"""Tests for the API client."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from vision_desktop_automation.api_client import ApiClient, Post


class TestPost:
    """Tests for the Post dataclass."""

    def test_format_content(self) -> None:
        post = Post(id=1, user_id=1, title="Hello", body="World")
        assert post.format_content() == "Title: Hello\n\nWorld"

    def test_format_content_multiline(self) -> None:
        post = Post(id=2, user_id=1, title="Test", body="Line1\nLine2\nLine3")
        result = post.format_content()
        assert result.startswith("Title: Test\n\n")
        assert "Line1\nLine2\nLine3" in result


class TestApiClient:
    """Tests for the ApiClient."""

    def test_fetch_posts_success(self) -> None:
        client = ApiClient()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 1, "userId": 1, "title": "Test", "body": "Body"},
            {"id": 2, "userId": 1, "title": "Test2", "body": "Body2"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_response):
            posts = client.fetch_posts(count=2)
            assert len(posts) == 2
            assert posts[0].id == 1
            assert posts[0].title == "Test"

    def test_fetch_posts_count_limit(self) -> None:
        client = ApiClient()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": i, "userId": 1, "title": f"Post {i}", "body": f"Body {i}"}
            for i in range(1, 20)
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_response):
            posts = client.fetch_posts(count=5)
            assert len(posts) == 5

    def test_fetch_posts_malformed_data(self) -> None:
        client = ApiClient()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 1, "userId": 1, "title": "Good", "body": "OK"},
            {"id": 2},  # Missing fields
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_response):
            posts = client.fetch_posts(count=2)
            assert len(posts) == 1

    def test_fetch_posts_connection_error(self) -> None:
        import requests

        client = ApiClient()
        with patch.object(
            client.session, "get", side_effect=requests.ConnectionError("fail")
        ):
            with pytest.raises(ConnectionError):
                client.fetch_posts()
