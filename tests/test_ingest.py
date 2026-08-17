"""Tests for the asset ingest script (raw model files -> manifest)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _make_pack(tmp_path: Path, pack: str, filenames: list) -> Path:
    raw = tmp_path / "raw"
    d = raw / pack
    d.mkdir(parents=True, exist_ok=True)
    for fn in filenames:
        (d / fn).write_text("placeholder")
    return raw


def _ingest(tmp_path: Path, raw: Path) -> dict:
    out = tmp_path / "manifest.json"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "ingest_assets.py"),
         "--raw-dir", str(raw), "--out", str(out)],
        capture_output=True, text=True, cwd=repo,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text())


def test_same_model_multiple_formats_merges(tmp_path):
    """tree.glb and tree.obj are one model, not two."""
    raw = _make_pack(tmp_path, "kenney_test", ["tree.glb", "tree.obj"])
    manifest = _ingest(tmp_path, raw)
    assert manifest["asset_count"] == 1
    assert manifest["assets"][0]["format"] == ".glb"   # preferred format wins


def test_distinct_models_kept_as_variants(tmp_path):
    """Different models that clean to the same name must not be merged.

    Stripping random IDs made 'House_k6tP5nFuD2' and 'House_vZ1ClbWmSx' both
    become 'house'; merging them cost 17 distinct building models and would
    have made every house in a village identical.
    """
    raw = _make_pack(tmp_path, "quaternius_test",
                     ["House_k6t.glb", "House_p5n.glb", "House_z1x.glb"])
    manifest = _ingest(tmp_path, raw)

    assert manifest["asset_count"] == 3
    names = {a["name"] for a in manifest["assets"]}
    assert names == {"house"}                       # same searchable name
    ids = {a["id"] for a in manifest["assets"]}
    assert len(ids) == 3                            # distinct identities
    files = {a["file"] for a in manifest["assets"]}
    assert len(files) == 3                          # distinct meshes


def test_single_model_has_no_variant_suffix(tmp_path):
    raw = _make_pack(tmp_path, "kenney_test", ["windmill.glb"])
    manifest = _ingest(tmp_path, raw)
    asset = manifest["assets"][0]
    assert asset["variant"] is None
    assert "__v" not in asset["id"]


def test_manifest_has_required_fields(tmp_path):
    raw = _make_pack(tmp_path, "kenney_test", ["tree.glb", "rock_smallA.glb"])
    manifest = _ingest(tmp_path, raw)
    for a in manifest["assets"]:
        for field in ("id", "name", "category", "pack", "file",
                      "radius", "height", "modular", "tags"):
            assert field in a, f"missing {field}"
        assert a["radius"] > 0
