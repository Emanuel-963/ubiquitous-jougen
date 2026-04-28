"""Tests for Phase 6 — Settings panel + auto-update flow."""
from __future__ import annotations

import json
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from src.config import PipelineConfig
from src.updater import (
    ReleaseInfo,
    _parse_version,
    check_for_updates,
    download_release,
    extract_release,
    get_latest_release,
)

# ── _parse_version ────────────────────────────────────────────────────


def test_parse_version_with_v_prefix():
    assert _parse_version("v1.2.3") == (1, 2, 3)


def test_parse_version_without_prefix():
    assert _parse_version("0.3.0") == (0, 3, 0)


def test_parse_version_empty():
    assert _parse_version("") == (0,)


def test_parse_version_invalid():
    assert _parse_version("latest") == (0,)


# ── ReleaseInfo ──────────────────────────────────────────────────────


def _make_release(tag: str) -> ReleaseInfo:
    return ReleaseInfo(
        tag=tag,
        html_url=f"https://github.com/x/y/releases/tag/{tag}",
        zipball_url=f"https://api.github.com/repos/x/y/zipball/{tag}",
        body="## Changelog\n- fix: something",
    )


def test_release_info_is_newer_true():
    """Tag newer than local version returns True."""
    release = _make_release("v999.0.0")
    assert release.is_newer_than_local() is True


def test_release_info_is_newer_false():
    """Tag older than local version returns False."""
    release = _make_release("v0.0.1")
    assert release.is_newer_than_local() is False


def test_release_info_same_version():
    """Same tag as local version is not 'newer'."""
    from src import __version__ as _local

    release = _make_release(f"v{_local}")
    assert release.is_newer_than_local() is False


# ── get_latest_release ────────────────────────────────────────────────


def test_get_latest_release_returns_release_info():
    payload = {
        "tag_name": "v9.9.9",
        "html_url": "https://github.com/x/y/releases/tag/v9.9.9",
        "zipball_url": "https://api.github.com/repos/x/y/zipball/v9.9.9",
        "body": "notes",
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        info = get_latest_release()

    assert info is not None
    assert info.tag == "v9.9.9"
    assert info.html_url == payload["html_url"]
    assert info.zipball_url == payload["zipball_url"]
    assert info.body == "notes"


def test_get_latest_release_network_error_returns_none():
    with patch("urllib.request.urlopen", side_effect=OSError("no network")):
        result = get_latest_release()
    assert result is None


def test_get_latest_release_bad_json_returns_none():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"not-json"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = get_latest_release()
    assert result is None


# ── check_for_updates ─────────────────────────────────────────────────


def test_check_for_updates_returns_message_when_newer():
    release = _make_release("v999.0.0")
    with patch("src.updater.get_latest_release", return_value=release):
        msg = check_for_updates()
    assert msg is not None
    assert "v999.0.0" in msg


def test_check_for_updates_returns_none_when_up_to_date():
    release = _make_release("v0.0.1")
    with patch("src.updater.get_latest_release", return_value=release):
        msg = check_for_updates()
    assert msg is None


def test_check_for_updates_returns_none_on_error():
    with patch("src.updater.get_latest_release", return_value=None):
        msg = check_for_updates()
    assert msg is None


# ── download_release ──────────────────────────────────────────────────


def test_download_release_writes_file(tmp_path):
    fake_content = b"PK\x03\x04" + b"\x00" * 100  # fake zip header
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": str(len(fake_content))}
    mock_resp.read.side_effect = [fake_content, b""]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    release = _make_release("v1.0.0")
    dest = tmp_path / "update.zip"

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = download_release(release, dest)

    assert result == dest
    assert dest.exists()
    assert dest.read_bytes() == fake_content


def test_download_release_calls_progress_callback(tmp_path):
    fake_content = b"x" * 200
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "200"}
    mock_resp.read.side_effect = [fake_content, b""]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    release = _make_release("v1.0.0")
    calls: list = []

    with patch("urllib.request.urlopen", return_value=mock_resp):
        download_release(
            release, tmp_path / "u.zip", on_progress=lambda d, t: calls.append((d, t))
        )

    assert len(calls) >= 1
    downloaded, total = calls[-1]
    assert downloaded == 200
    assert total == 200


def test_download_release_creates_parent_dirs(tmp_path):
    fake_content = b"data"
    mock_resp = MagicMock()
    mock_resp.headers = {}
    mock_resp.read.side_effect = [fake_content, b""]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    release = _make_release("v1.0.0")
    dest = tmp_path / "deep" / "nested" / "update.zip"

    with patch("urllib.request.urlopen", return_value=mock_resp):
        download_release(release, dest)

    assert dest.exists()


# ── extract_release ───────────────────────────────────────────────────


def test_extract_release_extracts_zip(tmp_path):
    zip_path = tmp_path / "update.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("myrepo-abc123/src/__init__.py", "# hello")
        zf.writestr("myrepo-abc123/README.md", "readme")

    result = extract_release(zip_path, tmp_path / "extracted")

    assert result.name == "myrepo-abc123"
    assert (result / "src" / "__init__.py").exists()
    assert (result / "README.md").exists()


def test_extract_release_creates_dest_dir(tmp_path):
    zip_path = tmp_path / "update.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("pkg/file.txt", "hi")

    dest = tmp_path / "new_folder" / "inner"
    extract_release(zip_path, dest)
    assert dest.exists()


# ── PipelineConfig round-trip ─────────────────────────────────────────


def test_pipeline_config_to_json_from_json_round_trip(tmp_path):
    cfg = PipelineConfig.default()
    cfg.scan_rate = 2.5
    cfg.drt_lambda = 1e-4
    cfg.drt_n_taus = 80
    cfg.llm_provider = "openai"
    cfg.llm_model = "gpt-4"

    path = tmp_path / "config.json"
    cfg.to_json(path)

    loaded = PipelineConfig.from_json(path)
    assert loaded.scan_rate == pytest.approx(2.5)
    assert loaded.drt_lambda == pytest.approx(1e-4)
    assert loaded.drt_n_taus == 80
    assert loaded.llm_provider == "openai"
    assert loaded.llm_model == "gpt-4"


def test_pipeline_config_from_json_unknown_keys_ignored(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"scan_rate": 3.0, "unknown_key": "ignore_me"}))

    cfg = PipelineConfig.from_json(path)
    assert cfg.scan_rate == pytest.approx(3.0)


def test_pipeline_config_from_json_safe_fallback(tmp_path):
    bad_path = tmp_path / "nonexistent.json"
    cfg = PipelineConfig.from_json_safe(bad_path)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.scan_rate == PipelineConfig.default().scan_rate


def test_pipeline_config_to_dict_tuples_become_lists():
    cfg = PipelineConfig.default()
    d = cfg.to_dict()
    assert isinstance(d["required_columns"], list)
    assert isinstance(d["circuit_multi_seed_scales"], list)
