"""JSONPlaceholder API client."""

from __future__ import annotations

from dataclasses import dataclass

import requests
from loguru import logger


@dataclass
class Post:
    """Represents a blog post from JSONPlaceholder."""

    id: int
    user_id: int
    title: str
    body: str

    def format_content(self) -> str:
        """Format the post for writing to Notepad."""
        return f"Title: {self.title}\n\n{self.body}"


class ApiClient:
    """Client for the JSONPlaceholder API."""

    def __init__(self, base_url: str = "https://jsonplaceholder.typicode.com") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def fetch_posts(self, count: int = 10) -> list[Post]:
        """Fetch blog posts from the API.

        Args:
            count: Number of posts to fetch (max 100).

        Returns:
            List of Post objects.

        Raises:
            ConnectionError: If the API is unavailable.
            ValueError: If the response is invalid.
        """
        url = f"{self.base_url}/posts"
        logger.info(f"Fetching {count} posts from {url}")

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.ConnectionError as e:
            logger.error(f"API connection failed: {e}")
            raise ConnectionError(
                f"Cannot reach JSONPlaceholder API at {url}. "
                "Check your internet connection."
            ) from e
        except requests.HTTPError as e:
            logger.error(f"API HTTP error: {e}")
            raise ConnectionError(f"API returned error: {e}") from e

        data = response.json()
        if not isinstance(data, list):
            raise ValueError(f"Unexpected API response type: {type(data)}")

        posts = []
        for item in data[:count]:
            try:
                posts.append(
                    Post(
                        id=item["id"],
                        user_id=item["userId"],
                        title=item["title"],
                        body=item["body"],
                    )
                )
            except KeyError as e:
                logger.warning(f"Skipping malformed post: missing key {e}")

        logger.info(f"Fetched {len(posts)} posts")
        return posts

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()
