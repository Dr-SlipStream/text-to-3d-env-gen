"""Tests for the closed loop: settings in the prompt, the bot playtest,
candidate selection, the web app, and the layout fixes the playtest drove.
"""

import json
import math
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import trimesh

from src import prompt_options
from src.asset_resolution import resolve_scene
from src.crossings import connect_banks, plan_crossings
from src.layout import layout_scene, settlement_radius, validate
from src.playtest import WEIGHTS, playtest
from src.prompt_decomposition import decompose
from src.schema import TerrainFeature
from src.terrain import Terrain


# --- settings in the sentence -------------------------------------------------

@pytest.mark.parametrize("prompt,clean,check", [
    ("a village, best of 5", "a village", lambda o: o.candidates == 5),
    ("a camp, variation 7", "a camp", lambda o: o.seed == 7),
    ("an outpost for unity", "an outpost", lambda o: o.unity),
    ("a forest, quick draft", "a forest", lambda o: o.offline),
    ("a 200m town by a lake", "a town by a lake",
     lambda o: o.size == "large"),
    ("a fort, simple layout", "a fort", lambda o: o.mode == "rules"),
    ("a farm, no bot", "a farm",
     lambda o: not o.playtest and o.candidates == 1),
])
def test_settings_are_read_and_removed(prompt, clean, check):
    text, opts = prompt_options.parse(prompt)
    assert text == clean
    assert check(opts)
    assert opts.notes, "nothing reported back to the user"


def test_descriptive_settings_stay_in_the_prompt():
    """"dense" describes the place too; the model should still hear it."""
    text, opts = prompt_options.parse("a dense forest camp")
    assert "dense" in text and opts.density > 1


def test_plain_prompt_gets_defaults():
    text, opts = prompt_options.parse("a medieval village beside a river")
    assert text == "a medieval village beside a river"
    assert opts.mode == "graph" and opts.candidates == 3 and opts.playtest
    assert opts.notes == []


def test_number_words_work():
    assert prompt_options.parse("a camp, best of four")[1].candidates == 4


# --- the layout fixes ----------------------------------------------------------

def _river_scene(index, seed, prompt="a medieval village beside a river"):
    spec = decompose(prompt, force_fallback=True)
    spec.terrain.size = "large"
    resolved = resolve_scene(spec, index)
    t = Terrain.from_spec(spec, seed=seed)
    t.apply_features(spec.features, seed=seed)
    return layout_scene(resolved, t, seed=seed), t, resolved


@pytest.mark.parametrize("seed", range(5))
def test_river_road_is_bridged_not_dead_ended(index, seed):
    """The road used to stop at the bank: zero crossings scored best."""
    scene, terrain, _ = _river_scene(index, seed)
    assert terrain.features.bridges, "road never crosses the river"


def test_sea_trims_but_river_does_not():
    t = Terrain.generate("grassland", 120, seed=4)
    t.apply_features([TerrainFeature(type="river")], seed=4)
    assert not t.features.sea_mask.any()
    c = Terrain.generate("beach_sand", 120, seed=4)
    c.apply_features([TerrainFeature(type="coast")], seed=4)
    assert c.features.sea_mask.any()


@pytest.mark.parametrize("seed", range(6))
def test_no_large_bank_is_cut_off(index, seed):
    """A river bending through two map edges used to strand up to 46%."""
    scene, _, _ = _river_scene(index, seed)
    report = playtest(scene, train_bot=False)
    assert report.metric("reachability").value >= 90


def test_village_size_follows_its_buildings():
    """Six houses and thirty houses should not get the same footprint."""
    from types import SimpleNamespace as O

    def village(houses):
        return [O(category="building", radius=4.0, quantity=houses,
                  resolved=True),
                O(category="structure", radius=1.5, quantity=120,
                  resolved=True)]          # fences: no plots of their own

    few = settlement_radius(village(6), 200, "medieval_village")
    many = settlement_radius(village(30), 200, "medieval_village")
    assert few < many <= 0.30 * 200
    assert few < 30, "six houses spread over a field"


def test_connect_banks_is_a_no_op_without_water():
    t = Terrain.generate("grassland", 120, seed=1)
    assert connect_banks(t, [(0, 60), (120, 60)]) == []


# --- the playtest ----------------------------------------------------------------

@pytest.fixture(scope="module")
def played(index):
    scene, _, _ = _river_scene(index, 1)
    return scene, playtest(scene, episodes=150, seeds=(0,))


def test_playtest_reports_every_metric(played):
    _, report = played
    keys = {m.key for m in report.metrics}
    assert keys == set(WEIGHTS)
    for m in report.metrics:
        assert 0 <= m.score <= 100, m
        assert m.note


def test_flow_score_is_a_weighted_mean(played):
    _, report = played
    expect = sum(WEIGHTS[m.key] * m.score for m in report.metrics) \
        / sum(WEIGHTS.values())
    assert report.flow_score == pytest.approx(expect)


def test_bot_learns_the_route(played):
    _, report = played
    assert report.connected
    assert report.metric("learnability").score > 50
    # The bot's route ends at the goal and is no shorter than optimal.
    assert tuple(report.bot_route[-1]) == tuple(report.goal)
    assert len(report.bot_route) - 1 >= report.optimal_steps
    # And it got better with practice.
    assert report.steps_curve[-1] < report.steps_curve[0]


def test_unreachable_goal_caps_the_score(played):
    scene, _ = played
    from src.layout import PlacedInstance
    # Wall off the goal end of the road with a ring of big rocks.
    gx, gz = scene.path[-1]
    blocked = list(scene.instances)
    for k in range(16):
        a = k / 16 * 2 * math.pi
        blocked.append(PlacedInstance(
            name="rock", category="rock", asset_name="r", asset_file="",
            position=(gx + 14 * math.cos(a), 0, gz + 14 * math.sin(a)),
            rotation_y=0, scale=1, radius=6, triangles=0))
    original = scene.instances
    scene.instances = blocked
    try:
        report = playtest(scene, train_bot=False)
    finally:
        scene.instances = original
    if report.connected:
        pytest.skip("goal still reachable round the ring")
    assert report.flow_score <= 30


def test_report_serialises(played):
    _, report = played
    data = json.loads(json.dumps(report.to_dict()))
    assert len(data["grid"]) == 40 and data["grade"] in "ABCDF"


# --- the pipeline and the web app ------------------------------------------------

@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("lib")
    model = root / "box.glb"
    mesh = trimesh.creation.box(extents=(1, 2, 1))
    mesh.apply_translation([0, 1, 0])
    mesh.export(model)
    rows = [("house", "building", 1.0, 1.6), ("tree", "vegetation", 0.4, 3.0),
            ("rock", "rock", 0.3, 0.4), ("barrel", "prop", 0.2, 0.5),
            ("fence", "structure", 0.5, 0.6),
            ("torch", "light_source", 0.1, 1.0),
            ("bridge wood", "structure", 1.0, 0.5)]
    path = root / "manifest.json"
    path.write_text(json.dumps({
        "version": 1, "generated": "test", "asset_count": len(rows),
        "assets": [{
            "id": f"t/{n}", "name": n, "category": c, "pack": "t",
            "file": str(model), "format": ".glb", "radius": r, "height": h,
            "measured": True, "triangles": 12, "modular": False,
            "variant": None, "theme_hints": [], "tags": [n, c],
        } for n, c, r, h in rows],
    }), encoding="utf-8")
    return path


def test_pipeline_runs_the_whole_loop(manifest, tmp_path):
    from src.pipeline import run
    result = run("a village beside a river, best of 2, quick draft",
                 manifest=manifest, out_root=tmp_path, log=lambda m: None)
    out = Path(result["out_dir"])
    for name in ("index.html", "scene.json", "playtest.json",
                 "summary.json", "lighting.json"):
        assert (out / name).exists(), name
    assert len(result["candidates"]) == 2
    best = max(result["candidates"], key=lambda c: (c["valid"], c["flow"]))
    assert result["chosen_seed"] == best["seed"]
    assert "playtest" in result
    # The setting phrases never reached the parser.
    assert "best of" not in result["clean_prompt"]


def test_pipeline_is_reproducible(manifest, tmp_path):
    from src.pipeline import stable_seed
    assert stable_seed("A village ") == stable_seed("a village")


def test_web_app_generates_a_scene(manifest, tmp_path):
    import argparse
    from http.server import ThreadingHTTPServer
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "serve", Path(__file__).resolve().parents[1] / "scripts" / "serve.py")
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)

    config = argparse.Namespace(out=tmp_path, manifest=manifest,
                                offline=True, ready=threading.Event())
    config.ready.set()
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 serve.make_handler(config,
                                                    serve.Jobs(config)))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, r.read()

    try:
        status, page = get("/")
        assert status == 200 and b"Generate scene" in page

        _, body = get("/api/options?prompt=a%20camp%2C%20best%20of%204")
        assert json.loads(body)["options"]["candidates"] == 4

        req = urllib.request.Request(
            base + "/api/generate", method="POST",
            data=json.dumps({"prompt": "a forest camp, no bot"}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            job = json.loads(r.read())["job"]

        for _ in range(240):
            _, body = get(f"/api/jobs/{job}")
            state = json.loads(body)
            if state["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        assert state["status"] == "done", state.get("error")

        slug = state["result"]["slug"]
        _, body = get("/api/scenes")
        assert any(s["slug"] == slug for s in json.loads(body))
        status, _ = get(f"/scenes/{slug}/index.html")
        assert status == 200

        # Nothing outside the output folder is reachable.
        with pytest.raises(urllib.error.HTTPError):
            get("/scenes/../../README.md")
    finally:
        server.shutdown()
