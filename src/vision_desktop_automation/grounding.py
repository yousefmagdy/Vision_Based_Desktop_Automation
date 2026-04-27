"""Visual grounding module using Florence-2 with cascaded search.

Implements a flexible, open-vocabulary visual grounding system inspired by
ScreenSeekeR (ScreenSpot-Pro, arXiv:2504.07981). The system uses Microsoft's
Florence-2 vision-language model to locate arbitrary UI elements on screen
without requiring pre-captured template images.

Architecture:
    1. OCR-based text search: Use Florence-2 OCR to find icon labels on screen
       (most reliable for desktop icons with text labels)
    2. Phrase grounding: Run Florence-2 phrase grounding as secondary method
    3. Cascaded search: Split into overlapping grid regions if full-image fails
    4. OCR verification: Verify detections by checking nearby text matches
    5. Confidence scoring & NMS: Aggregate detections, apply non-maximum suppression
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

from .config import Config


@dataclass
class Detection:
    """A detected UI element on screen."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in screen coords
    confidence: float
    label: str
    center: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class VisualGrounder:
    """Florence-2 based visual grounding with cascaded search.

    This implementation follows the ScreenSeekeR paradigm:
    - Uses a VLM (Florence-2) for open-vocabulary element detection
    - Implements cascaded cropping to handle high-resolution screens
    - Supports arbitrary text queries, making it generalizable to any icon/button
    - Uses OCR as the primary detection method for text-labeled UI elements
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._model = None
        self._processor = None
        self._model_loaded = False

    def load_model(self) -> None:
        """Load the Florence-2 model and processor."""
        if self._model_loaded:
            return

        logger.info(f"Loading model: {self.config.model_name} on {self.config.device}")
        start = time.time()

        try:
            import torch
            from transformers import AutoProcessor

            dtype = torch.float16 if self.config.device == "cuda" else torch.float32

            # Florence-2 has native support in transformers >=4.46 via
            # Florence2ForConditionalGeneration. The microsoft/Florence-2-large
            # repo's custom remote code is broken with transformers >=4.50
            # (Florence2LanguageConfig missing forced_bos_token_id).
            #
            # Strategy:
            #  1. Try native Florence2ForConditionalGeneration (works with transformers >=4.46)
            #  2. Try the florence-community re-upload (native-compatible weights)
            #  3. Fall back to AutoModelForCausalLM + trust_remote_code (older transformers)
            loaded = False
            strategies = [
                ("native", self.config.model_name, False),
                ("community", "florence-community/Florence-2-large", False),
                ("remote_code", self.config.model_name, True),
            ]

            for strategy_name, model_id, trust_remote in strategies:
                try:
                    if not trust_remote:
                        from transformers import Florence2ForConditionalGeneration

                        self._processor = AutoProcessor.from_pretrained(model_id)
                        self._model = Florence2ForConditionalGeneration.from_pretrained(
                            model_id,
                            torch_dtype=dtype,
                        ).to(self.config.device)
                    else:
                        from transformers import AutoModelForCausalLM

                        self._processor = AutoProcessor.from_pretrained(
                            model_id, trust_remote_code=True,
                        )
                        self._model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            trust_remote_code=True,
                            torch_dtype=dtype,
                        ).to(self.config.device)

                    loaded = True
                    logger.info(f"Model loaded via strategy '{strategy_name}' from '{model_id}'")
                    break
                except Exception as load_err:
                    logger.debug(
                        f"Strategy '{strategy_name}' (model={model_id}) failed: {load_err}"
                    )
                    continue

            if not loaded:
                raise RuntimeError(
                    "All loading strategies failed. See debug log for details. "
                    "Try: pip install transformers>=4.46"
                )

            self._model.eval()
            self._model_loaded = True
            elapsed = time.time() - start
            logger.info(f"Model loaded in {elapsed:.1f}s on {self.config.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(
                f"Could not load Florence-2 model '{self.config.model_name}'. "
                f"Ensure you have torch and transformers installed. Error: {e}"
            ) from e

    def _run_florence(
        self,
        image: Image.Image,
        task: str,
        query: str = "",
    ) -> dict:
        """Run a Florence-2 task and return parsed results.

        Args:
            image: Input image.
            task: Florence-2 task prompt (e.g., "<OCR_WITH_REGION>").
            query: Optional text query (appended to task prompt).

        Returns:
            Parsed result dict from the processor.
        """
        import torch

        prompt = task + query if query else task
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=3,
            )

        raw = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            raw,
            task=task,
            image_size=(image.width, image.height),
        )
        return parsed

    def _ocr_search(
        self,
        image: Image.Image,
        target: str,
    ) -> list[Detection]:
        """Find the target by searching for its text label via OCR.

        This is the most reliable method for desktop icons, which always
        have a text label beneath the icon image. We find the text bbox
        and then shift the click target upward to hit the icon itself.

        Args:
            image: Screenshot.
            target: Text to search for (e.g., "Notepad").

        Returns:
            List of Detection objects where matching text was found.
        """
        logger.debug(f"Running OCR search for '{target}'")
        target_lower = target.lower()

        parsed = self._run_florence(image, "<OCR_WITH_REGION>")

        detections = []
        if "<OCR_WITH_REGION>" in parsed:
            result = parsed["<OCR_WITH_REGION>"]
            labels = result.get("labels", [])
            # quad_boxes are [x1,y1, x2,y1, x2,y2, x1,y2] format (4 corners)
            quad_boxes = result.get("quad_boxes", [])

            for label_text, quad in zip(labels, quad_boxes):
                clean_label = label_text.strip().rstrip("</s>").strip()
                logger.debug(f"  OCR found: '{clean_label}' at quad={quad}")

                if target_lower in clean_label.lower():
                    # Convert quad to bbox
                    xs = [quad[i] for i in range(0, 8, 2)]
                    ys = [quad[i] for i in range(1, 8, 2)]
                    text_x1, text_y1 = min(xs), min(ys)
                    text_x2, text_y2 = max(xs), max(ys)
                    text_w = text_x2 - text_x1
                    text_h = text_y2 - text_y1

                    # Desktop icon layout: icon image is ABOVE the text label.
                    # The icon is roughly 32-48px tall, and the text is below it.
                    # We expand the bbox upward to include the icon image,
                    # and set the click target to be the icon center (above text).
                    icon_height_estimate = max(40, text_w)  # Icon is roughly square
                    icon_x1 = text_x1 - 5
                    icon_y1 = text_y1 - icon_height_estimate
                    icon_x2 = text_x2 + 5
                    icon_y2 = text_y2 + 5

                    # High confidence because we matched the exact text
                    detections.append(Detection(
                        bbox=(icon_x1, icon_y1, icon_x2, icon_y2),
                        confidence=0.95,
                        label=clean_label,
                    ))
                    logger.info(
                        f"  OCR match: '{clean_label}' -> icon bbox "
                        f"({icon_x1:.0f},{icon_y1:.0f})-({icon_x2:.0f},{icon_y2:.0f})"
                    )

        if not detections:
            logger.debug(f"OCR search found no match for '{target}'")

        return detections

    def _phrase_grounding_search(
        self,
        image: Image.Image,
        query: str,
    ) -> list[Detection]:
        """Run Florence-2 phrase grounding on the image.

        Args:
            image: Input image.
            query: Text query describing the target.

        Returns:
            List of Detection objects.
        """
        parsed = self._run_florence(
            image, "<CAPTION_TO_PHRASE_GROUNDING>", query
        )

        detections = []
        task_key = "<CAPTION_TO_PHRASE_GROUNDING>"
        if task_key in parsed:
            result = parsed[task_key]
            bboxes = result.get("bboxes", [])
            labels = result.get("labels", [])

            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = [float(v) for v in bbox]
                area = max(0, x2 - x1) * max(0, y2 - y1)
                img_area = image.width * image.height
                size_ratio = area / img_area if img_area > 0 else 0
                # Reasonable icon size: 0.05% to 2% of screen
                if 0.0001 < size_ratio < 0.05:
                    confidence = 0.5  # Lower than OCR; needs verification
                else:
                    confidence = 0.2

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    label=label,
                ))

        return detections

    def _open_vocabulary_search(
        self,
        image: Image.Image,
        query: str,
    ) -> list[Detection]:
        """Run open-vocabulary detection.

        Args:
            image: Input image.
            query: Text query.

        Returns:
            List of Detection objects.
        """
        parsed = self._run_florence(image, "<OPEN_VOCABULARY_DETECTION>", query)

        detections = []
        task_key = "<OPEN_VOCABULARY_DETECTION>"
        if task_key in parsed:
            result = parsed[task_key]
            bboxes = result.get("bboxes", [])
            labels = result.get("bboxes_labels", result.get("labels", []))

            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = [float(v) for v in bbox]
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=0.4,  # Lower confidence than OCR
                    label=str(label),
                ))

        return detections

    def _cascaded_ocr_search(
        self,
        image: Image.Image,
        target: str,
    ) -> list[Detection]:
        """Run OCR search on grid sub-regions for better text detection.

        Small text on high-res screens can be missed by full-image OCR.
        Splitting into regions and running OCR on each gives better recall.

        Args:
            image: Full screenshot.
            target: Text to search for.

        Returns:
            List of Detection objects in full-image coordinates.
        """
        rows, cols = self.config.grid_size
        overlap = self.config.crop_overlap
        img_w, img_h = image.size

        cell_w = img_w / cols
        cell_h = img_h / rows
        overlap_w = int(cell_w * overlap)
        overlap_h = int(cell_h * overlap)

        all_detections: list[Detection] = []

        for row in range(rows):
            for col in range(cols):
                x1 = max(0, int(col * cell_w) - overlap_w)
                y1 = max(0, int(row * cell_h) - overlap_h)
                x2 = min(img_w, int((col + 1) * cell_w) + overlap_w)
                y2 = min(img_h, int((row + 1) * cell_h) + overlap_h)

                region = image.crop((x1, y1, x2, y2))
                logger.debug(
                    f"Cascaded OCR region [{row},{col}]: "
                    f"({x1},{y1})-({x2},{y2}), size={region.size}"
                )

                region_dets = self._ocr_search(region, target)

                # Map coordinates back to full image
                for det in region_dets:
                    rx1, ry1, rx2, ry2 = det.bbox
                    mapped_bbox = (rx1 + x1, ry1 + y1, rx2 + x1, ry2 + y1)
                    all_detections.append(Detection(
                        bbox=mapped_bbox,
                        confidence=det.confidence * 0.9,  # Slight decay
                        label=det.label,
                    ))

        return all_detections

    def _verify_detection_with_ocr(
        self,
        image: Image.Image,
        detection: Detection,
        target: str,
        margin: int = 60,
    ) -> bool:
        """Verify a detection by running OCR on the region around it.

        Crops a region around the detection and checks if the target text
        appears nearby. This prevents false positives from phrase grounding.

        Args:
            image: Full screenshot.
            detection: The detection to verify.
            target: Expected text label.
            margin: Pixel margin around the detection bbox.

        Returns:
            True if the target text is found near the detection.
        """
        x1, y1, x2, y2 = detection.bbox
        crop_x1 = max(0, int(x1) - margin)
        crop_y1 = max(0, int(y1) - margin)
        crop_x2 = min(image.width, int(x2) + margin)
        crop_y2 = min(image.height, int(y2) + margin)

        region = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        parsed = self._run_florence(region, "<OCR>")

        ocr_text = parsed.get("<OCR>", "").lower()
        target_lower = target.lower()
        found = target_lower in ocr_text

        logger.debug(
            f"OCR verification for '{target}' near "
            f"({detection.center[0]},{detection.center[1]}): "
            f"text='{ocr_text[:80]}', match={found}"
        )
        return found

    def detect_all_icons(
        self,
        image: Image.Image,
    ) -> list[Detection]:
        """Detect ALL desktop icons via OCR (Bonus Challenge #1).

        Scans the entire desktop for text labels and returns detections
        for each one. Useful for selecting the correct icon when multiple
        similar icons exist (e.g., "Notepad" vs "Notepad++").

        Args:
            image: Screenshot of the desktop.

        Returns:
            List of all detected icon labels with their positions.
        """
        self.load_model()
        logger.info("Detecting all desktop icons via OCR...")

        parsed = self._run_florence(image, "<OCR_WITH_REGION>")

        icons = []
        if "<OCR_WITH_REGION>" in parsed:
            result = parsed["<OCR_WITH_REGION>"]
            labels = result.get("labels", [])
            quad_boxes = result.get("quad_boxes", [])

            for label_text, quad in zip(labels, quad_boxes):
                clean_label = label_text.strip().rstrip("</s>").strip()
                if not clean_label:
                    continue

                xs = [quad[i] for i in range(0, 8, 2)]
                ys = [quad[i] for i in range(1, 8, 2)]
                text_x1, text_y1 = min(xs), min(ys)
                text_x2, text_y2 = max(xs), max(ys)

                icon_height_estimate = max(40, text_x2 - text_x1)
                bbox = (
                    text_x1 - 5,
                    text_y1 - icon_height_estimate,
                    text_x2 + 5,
                    text_y2 + 5,
                )
                icons.append(Detection(
                    bbox=bbox,
                    confidence=0.9,
                    label=clean_label,
                ))

        logger.info(f"Found {len(icons)} icon labels: {[i.label for i in icons]}")
        return icons

    def select_best_match(
        self,
        detections: list[Detection],
        target: str,
    ) -> Detection | None:
        """Select the best matching detection from multiple candidates.

        Handles disambiguation between similar names (e.g., "Notepad" vs
        "Notepad++") using exact match preference and string similarity.

        Args:
            detections: All detected icons.
            target: The desired icon name.

        Returns:
            Best matching Detection, or None.
        """
        if not detections:
            return None

        target_lower = target.lower()

        # Priority 1: Exact match (case-insensitive)
        exact = [d for d in detections if d.label.lower() == target_lower]
        if exact:
            return max(exact, key=lambda d: d.confidence)

        # Priority 2: Target is a prefix of the label (e.g., "Notepad" in "Notepad.lnk")
        prefix = [d for d in detections if d.label.lower().startswith(target_lower)]
        if prefix:
            # Prefer shorter labels (more exact match)
            return min(prefix, key=lambda d: len(d.label))

        # Priority 3: Target appears anywhere in the label
        contains = [d for d in detections if target_lower in d.label.lower()]
        if contains:
            return min(contains, key=lambda d: len(d.label))

        return None

    @staticmethod
    def _nms(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
        """Non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            kept.append(best)

            remaining = []
            for det in sorted_dets:
                if _compute_iou(best.bbox, det.bbox) < iou_threshold:
                    remaining.append(det)
            sorted_dets = remaining

        return kept

    def ground(
        self,
        image: Image.Image,
        target: str,
        region_hint: Optional[str] = None,
    ) -> Detection | None:
        """Locate a target element on the screen.

        This is the main entry point. It implements the full grounding pipeline:
        1. OCR search: Find the target's text label (highest reliability)
        2. Cascaded OCR: Sub-region OCR if full-image OCR misses small text
        3. Phrase grounding + OCR verification: Fallback with verification
        4. NMS to deduplicate
        5. Return the best detection

        Args:
            image: Screenshot of the desktop.
            target: Natural-language description of what to find (e.g., "Notepad").
            region_hint: Optional hint about where to look (e.g., "bottom-right").

        Returns:
            Best Detection, or None if nothing found.
        """
        self.load_model()

        logger.info(f"Grounding target: '{target}'")
        start = time.time()

        # ── Phase 1: OCR-based text search (most reliable for labeled icons) ──
        detections = self._ocr_search(image, target)
        if detections:
            logger.info(f"Phase 1 (OCR): Found {len(detections)} matches")
        else:
            logger.debug("Phase 1 (OCR): No matches on full image")

            # ── Phase 2: Cascaded OCR on sub-regions ──
            detections = self._cascaded_ocr_search(image, target)
            if detections:
                logger.info(f"Phase 2 (Cascaded OCR): Found {len(detections)} matches")
            else:
                logger.debug("Phase 2 (Cascaded OCR): No matches")

                # ── Phase 3: Phrase grounding with OCR verification ──
                logger.debug("Phase 3: Trying phrase grounding with OCR verification...")
                queries = [
                    f"{target} icon",
                    f"{target} shortcut",
                    f"{target}",
                ]
                for query in queries:
                    raw_dets = self._phrase_grounding_search(image, query)
                    raw_dets.extend(self._open_vocabulary_search(image, query))

                    # Verify each detection with OCR
                    for det in raw_dets:
                        if self._verify_detection_with_ocr(image, det, target):
                            det.confidence = 0.85  # Boost verified detections
                            detections.append(det)
                            logger.info(
                                f"Phase 3: Verified detection at "
                                f"({det.center[0]},{det.center[1]})"
                            )

                    if detections:
                        break

        if not detections:
            elapsed = time.time() - start
            logger.warning(
                f"No verified detections found for '{target}' after {elapsed:.1f}s. "
                f"Ensure a '{target}' shortcut exists on the desktop."
            )
            return None

        # ── Phase 4: NMS ──
        detections = self._nms(detections)

        # Apply region hint bias if provided
        if region_hint and detections:
            detections = self._apply_region_bias(detections, region_hint, image.size)

        # Filter by confidence threshold
        detections = [
            d for d in detections
            if d.confidence >= self.config.confidence_threshold
        ]

        if not detections:
            elapsed = time.time() - start
            logger.warning(
                f"All detections below confidence threshold for '{target}' "
                f"after {elapsed:.1f}s"
            )
            return None

        best = detections[0]
        elapsed = time.time() - start
        logger.info(
            f"Grounded '{target}' at center=({best.center[0]}, {best.center[1]}), "
            f"confidence={best.confidence:.2f}, elapsed={elapsed:.1f}s"
        )
        return best

    def _apply_region_bias(
        self,
        detections: list[Detection],
        hint: str,
        image_size: tuple[int, int],
    ) -> list[Detection]:
        """Bias detection scores based on a region hint."""
        w, h = image_size
        hint = hint.lower()

        def bias_score(det: Detection) -> float:
            cx, cy = det.center
            bonus = 0.0
            if "top" in hint and cy < h / 2:
                bonus += 0.1
            if "bottom" in hint and cy > h / 2:
                bonus += 0.1
            if "left" in hint and cx < w / 2:
                bonus += 0.1
            if "right" in hint and cx > w / 2:
                bonus += 0.1
            if "center" in hint:
                dist = ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) ** 0.5
                max_dist = (w ** 2 + h ** 2) ** 0.5 / 2
                bonus += 0.1 * (1 - dist / max_dist)
            return det.confidence + bonus

        return sorted(detections, key=bias_score, reverse=True)


def _compute_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Compute intersection-over-union of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0
