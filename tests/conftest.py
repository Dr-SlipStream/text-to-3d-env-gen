"""Shared test fixtures.

Builds a synthetic asset manifest that mimics the naming conventions of real
CC0 packs, so retrieval can be tested without downloading several hundred MB
of models.
"""

import pytest

from src.asset_index import AssetIndex, HashingEmbedder
from src import asset_rules

# Filenames modelled on Kenney's actual naming style, per pack.
FAKE_PACKS = {
    "kenney_nature-kit": [
        "tree_default.glb", "tree_pineDefaultA.glb", "tree_oak.glb",
        "tree_palmShort.glb", "bush_small.glb", "bush_large.glb",
        "rock_smallA.glb", "rock_largeA.glb", "boulder_round.glb",
        "grass_leafs.glb", "flower_redA.glb", "mushroom_red.glb",
        "log_large.glb", "stump_old.glb", "cactus_short.glb",
        "bridge_wood.glb", "fence_simple.glb", "path_stone.glb",
    ],
    "kenney_survival-kit": [
        "tent_smallOpen.glb", "tent_detailedClosed.glb",
        "campfire_stones.glb", "campfire_logs.glb",
        "crate_default.glb", "barrel_open.glb", "bag_open.glb",
        "lantern_hanging.glb", "torch_lit.glb",
        "structure_platform.glb", "fence_wood.glb",
    ],
    "kenney_medieval-town": [
        "house_small.glb", "house_large.glb", "cottage_thatched.glb",
        "building_tavern.glb", "building_smithy.glb", "windmill_tall.glb",
        "tower_watch.glb", "church_stone.glb", "market_stall.glb",
        "wall_stoneA.glb", "gate_wooden.glb", "well_stone.glb",
        "cart_wooden.glb", "anvil_iron.glb", "barrel_wine.glb",
        "sign_hanging.glb", "torch_wall.glb",
    ],
    "kenney_space-kit": [
        "structure_dome.glb", "structure_hangar.glb", "building_station.glb",
        "platform_large.glb", "container_metal.glb", "canister_fuel.glb",
        "antenna_tall.glb", "crystal_glowA.glb", "rock_alienA.glb",
        "neon_signA.glb", "spotlight_floor.glb", "pipe_segment.glb",
    ],
}


def build_fake_assets():
    """Produce manifest-shaped asset dicts from the fake pack listing."""
    assets = []
    for pack, filenames in FAKE_PACKS.items():
        hints = asset_rules.theme_hints(pack)
        for fn in filenames:
            name = asset_rules.clean_name(fn)
            category = asset_rules.classify(name, pack)
            assets.append({
                "id": f"{pack}/{name}".replace(" ", "_"),
                "name": name,
                "category": category,
                "pack": pack,
                "file": f"data/asset_library/raw/{pack}/{fn}",
                "format": ".glb",
                "radius": 1.0,
                "height": 2.0,
                "measured": False,
                "theme_hints": hints,
                "tags": asset_rules.build_tags(name, pack, category),
            })
    return assets


@pytest.fixture(scope="session")
def fake_assets():
    return build_fake_assets()


@pytest.fixture(scope="session")
def index(fake_assets):
    """Index built with the offline hashing embedder, so tests need no
    downloads and give identical results on every machine."""
    return AssetIndex(fake_assets, embedder=HashingEmbedder())
