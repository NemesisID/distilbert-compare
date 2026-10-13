"""Uji berkas deployment: keberadaan, isi kunci, dan XML yang valid."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
WINSERVICE = DEPLOY / "windows-service"

POWERSHELL_FILES = [
    "run_api.ps1",
    "healthcheck.ps1",
    "build_artifacts.ps1",
    "backup_db.ps1",
]


def test_deploy_files_exist():
    for name in POWERSHELL_FILES + ["README.md", "env.production.example"]:
        assert (DEPLOY / name).is_file(), f"hilang: deploy/{name}"


def test_windows_service_templates_exist():
    assert (WINSERVICE / "README.md").is_file()
    assert (WINSERVICE / "install_service.nssm.ps1.example").is_file()
    assert (WINSERVICE / "chatbot-api.winsw.xml.example").is_file()


def test_run_api_targets_uvicorn_and_preflight():
    text = (DEPLOY / "run_api.ps1").read_text(encoding="utf-8")
    assert "uvicorn" in text
    assert "api.main:app" in text
    assert "scripts.preflight" in text


def test_build_artifacts_seeds_and_indexes():
    text = (DEPLOY / "build_artifacts.ps1").read_text(encoding="utf-8")
    assert "scripts.seed_responses" in text
    assert "rag.index" in text


def test_healthcheck_targets_health_endpoint():
    text = (DEPLOY / "healthcheck.ps1").read_text(encoding="utf-8")
    assert "/health" in text


def test_winsw_example_is_valid_xml_and_local():
    path = WINSERVICE / "chatbot-api.winsw.xml.example"
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    assert root.tag == "service"
    arguments = root.findtext("arguments") or ""
    assert "api.main:app" in arguments
    assert "127.0.0.1" in arguments
    assert (root.findtext("executable") or "").endswith("python.exe")
    assert root.findtext("workingdirectory")


def test_templates_do_not_contain_real_secret():
    """Contoh deployment hanya memakai placeholder, bukan kunci sungguhan."""
    text = (DEPLOY / "env.production.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("CHATBOT_API_KEY="):
            value = line.split("=", 1)[1]
            assert "GANTI" in value or value == ""
