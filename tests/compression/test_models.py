"""Tests for P1-S7: platform-aware model cache path resolution.

The cache directory must resolve correctly across Windows, Android
(XDG_DATA_HOME set by the KMP bridge), and Linux/macOS, with an explicit
``ALICE_MODEL_CACHE`` override always winning. ``sys.platform`` and the
relevant env vars are patched so each branch is exercised regardless of the
host the suite runs on.
"""

from __future__ import annotations

from pathlib import Path

from backend.workers.app.compression import models as models_mod


_ENV_VARS = ("ALICE_MODEL_CACHE", "LOCALAPPDATA", "XDG_DATA_HOME")


def _clear_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_alice_model_cache_env_wins(monkeypatch):
    # Override is set AND a platform var is set -- override must win.
    _clear_env(monkeypatch)
    monkeypatch.setattr(models_mod.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\x\AppData\Local")
    monkeypatch.setenv("ALICE_MODEL_CACHE", "/tmp/test-models")

    assert models_mod._cache_dir() == Path("/tmp/test-models")


def test_windows_path_resolution(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(models_mod.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\x\AppData\Local")

    expected = Path(r"C:\Users\x\AppData\Local") / "project-alice" / "models"
    assert models_mod._cache_dir() == expected


def test_xdg_data_home_resolution(monkeypatch):
    # Android simulation: not win32, XDG_DATA_HOME points at Context.filesDir.
    _clear_env(monkeypatch)
    monkeypatch.setattr(models_mod.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/data/user/0/com.alice.app/files")

    expected = Path("/data/user/0/com.alice.app/files") / "project-alice" / "models"
    assert models_mod._cache_dir() == expected


def test_linux_fallback(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(models_mod.sys, "platform", "linux")

    expected = Path.home() / ".local" / "share" / "project-alice" / "models"
    assert models_mod._cache_dir() == expected
