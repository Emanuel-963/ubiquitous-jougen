"""Tests for src/updater.py — auto-update checker."""
from unittest.mock import patch
from src.updater import check_for_updates, _parse_version


class TestParseVersion:
    def test_normal_tag(self):
        assert _parse_version("v0.1.0") == (0, 1, 0)

    def test_no_prefix(self):
        assert _parse_version("1.2.3") == (1, 2, 3)

    def test_garbage_returns_zero(self):
        assert _parse_version("nope") == (0,)


class TestCheckForUpdates:
    def test_returns_none_on_network_error(self):
        """Network failures must not propagate — returns None silently."""
        with patch("src.updater.urllib.request.urlopen", side_effect=OSError):
            assert check_for_updates() is None

    def test_returns_message_when_newer(self):
        fake_json = b'{"tag_name":"v99.0.0","html_url":"https://example.com"}'
        mock_resp = type("R", (), {
            "read": lambda self: fake_json,
            "__enter__": lambda self: self,
            "__exit__": lambda *a: None,
        })()
        with patch("src.updater.urllib.request.urlopen", return_value=mock_resp):
            msg = check_for_updates()
            assert msg is not None
            assert "v99.0.0" in msg

    def test_returns_none_when_up_to_date(self):
        fake_json = b'{"tag_name":"v0.0.1","html_url":"https://example.com"}'
        mock_resp = type("R", (), {
            "read": lambda self: fake_json,
            "__enter__": lambda self: self,
            "__exit__": lambda *a: None,
        })()
        with patch("src.updater.urllib.request.urlopen", return_value=mock_resp):
            assert check_for_updates() is None
