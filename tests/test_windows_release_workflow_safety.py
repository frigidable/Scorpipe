"""Headless workflow safety contracts; not a replacement for Windows runner validation."""
from pathlib import Path

import yaml

WORKFLOW = (Path(__file__).resolve().parents[1] /
            ".github/workflows/windows_release.yml")


def _load() -> dict:
    # PyYAML BaseLoader keeps the YAML 1.2 'on' event key a string.
    data = yaml.load(WORKFLOW.read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)
    assert isinstance(data, dict)
    return data


def _step(job: dict, name: str) -> dict:
    steps = [s for s in job["steps"] if s.get("name") == name]
    assert len(steps) == 1, f"Expected exactly one {name!r} step"
    return steps[0]


def test_windows_publish_only_for_published_release_not_manual_dispatch() -> None:
    data = _load()
    assert set(data["on"]) == {"release", "workflow_dispatch", "pull_request"}
    assert data["on"]["release"]["types"] == ["published"]
    pr = data["on"]["pull_request"]
    assert pr["branches"] == ["integration/scorpipe-vnext-20261111"]
    assert set(pr["types"]) == {"opened", "reopened", "synchronize", "ready_for_review"}
    assert set(pr["paths"]) == {
        ".github/workflows/windows_release.yml",
        "packaging/windows/**",
        "tests/test_windows_release_workflow_safety.py",
    }
    jobs = data["jobs"]
    assert set(jobs) == {"build_check", "build", "artifact_contract", "publish"}
    assert jobs["build"]["needs"] == ["build_check"]
    assert jobs["artifact_contract"]["needs"] == ["build"]
    assert jobs["artifact_contract"]["if"] == "github.event_name == 'pull_request'"
    assert jobs["artifact_contract"]["permissions"] == {"contents": "read"}
    assert jobs["publish"]["needs"] == ["build"]
    assert jobs["publish"]["if"] == "github.event_name == 'release'"
    assert jobs["publish"]["permissions"] == {
        "actions": "read",
        "contents": "write",
        "id-token": "write",
        "attestations": "write",
    }
    assert data["permissions"] == {"contents": "read"}
    publish = _step(jobs["publish"], "Publish assets to GitHub Release")
    assert "gh release upload" in publish["run"]
    assert "--repo" in publish["run"]


def test_windows_artifact_roundtrip_stays_archived_and_same_name() -> None:
    jobs = _load()["jobs"]
    upload = _step(jobs["build"], "Upload release artifacts")
    download = _step(jobs["publish"], "Download release artifacts")
    qa_download = _step(jobs["artifact_contract"],
                        "Download Windows artifacts for nonpublishing verification")
    assert qa_download["uses"] == download["uses"]
    assert qa_download["with"] == download["with"]
    verify = _step(jobs["artifact_contract"],
                   "Verify Windows installer, portable bundle and checksums")
    assert "sha256sum --check --strict SHA256SUMS.txt" in verify["run"]
    assert "unzip -t" in verify["run"]
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert download["uses"].startswith("actions/download-artifact@")
    assert upload["with"]["name"] == download["with"]["name"] == "windows-release-assets"
    assert upload["with"]["path"] == "release/*"
    assert upload["with"]["archive"] == "true"
    assert upload["with"]["if-no-files-found"] == "error"
    assert download["with"]["path"] == "release"
    assert _step(jobs["build"], "Package artifacts + checksums")["run"].find(
        "SHA256SUMS.txt") >= 0


def test_sbom_paths_match_attestation_and_checksums() -> None:
    publish = _load()["jobs"]["publish"]
    for name in ("installer", "portable bundle", "wheel", "sdist"):
        generated = _step(publish, f"Generate SBOM ({name})")
        attested = _step(publish, f"Attest SBOM ({name})")
        assert generated["with"]["output-file"] == attested["with"]["sbom-path"]
        assert generated["with"]["format"] == "spdx-json"
        assert generated["with"]["upload-artifact"] == "false"
        assert generated["with"]["upload-release-assets"] == "false"
        assert generated["uses"].startswith("anchore/sbom-action@")
        assert attested["uses"].startswith("actions/attest-sbom@")
    checksum = _step(publish, "Add SBOMs to checksums")["run"]
    assert "sha256sum" in checksum
    assert "SHA256SUMS.txt" in checksum
    provenance = _step(publish, "Attest build provenance")
    assert provenance["with"]["subject-checksums"] == "release/SHA256SUMS.txt"


def test_workflow_action_versions_are_pinned_not_moving_tags() -> None:
    import re

    data = _load()
    for job in data["jobs"].values():
        for step in job["steps"]:
            if "uses" not in step:
                continue
            action = step["uses"]
            assert re.fullmatch(r"[^@ ]+@[0-9a-f]{40}", action), action


def test_windows_build_version_uses_native_powershell_parser() -> None:
    """Prevent the real PR #213 Windows failure from nested python -c quoting."""
    run = _step(_load()["jobs"]["build"], "Resolve build version")["run"]
    assert 'Get-Content -LiteralPath "src/scorpio_pipe/version.py"' in run
    assert "[regex]::Match(" in run
    assert '$versionMatch.Groups["value"].Value' in run
    assert "if (-not $versionMatch.Success)" in run
    assert "python -c" not in run
    assert 'APP_VERSION=$ver' in run
