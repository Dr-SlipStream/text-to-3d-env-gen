"""
Getting a scene graph out of a prompt.

Two routes, same output:

  ask_for_graph()   the language model composes regions, groups and relations
                    directly. This is the point of the whole design -- a new
                    kind of place needs no code, because the model builds it
                    from the same ten relations.

  compile_graph()   a graph derived from the old flat specification, using
                    fixed rules. Used when the model is unavailable, and as
                    the comparison that shows what the model adds.

The compiler deliberately keeps its rules in one small place instead of
spread across the pipeline. They exist to keep the fallback working, not to
carry the system.
"""

from __future__ import annotations

from typing import List, Optional

from . import vocab
from .llm_client import LLMClient, LLMUnavailable
from .scene_graph import (RELATION_KINDS, REGION_KINDS, Group, Region,
                          Relation, SceneGraph)

# ---------------------------------------------------------------------------
# Route 1: the language model composes the graph
# ---------------------------------------------------------------------------

GRAPH_SYSTEM = f"""You describe video-game environments as a scene graph.

Reply with ONLY a JSON object of this shape:

{{
  "regions": [
    {{"id": "short_name", "kind": one of {REGION_KINDS}, "size": "small"|"medium"|"large"}}
  ],
  "groups": [
    {{
      "id": "short_name",
      "asset": "what to look for in a 3D model library, 1-3 lowercase words",
      "count": exact number, OR "density": number per 100 square metres,
      "relations": [
        {{"kind": one of {RELATION_KINDS}, "target": "an id from this graph",
          "distance": metres}}
      ]
    }}
  ]
}}

WHAT THE RELATIONS MEAN
  inside         somewhere within a region
  at_center_of   the focal point of a region -- a village well, a camp fire
  ring_around    evenly spaced at `distance` from a target
  cluster_near   loosely gathered within `distance` of a target
  along          following a line: a road, a riverbank
  on_shore_of    on the land beside water, within `distance` of the edge
  scatter_in     spread through a region, naturally clumped
  avoid          keep at least `distance` away from a target
  faces          turn to look at a target
  aligns_with    turn parallel to a line

RULES
- Every "target" must be an id you defined in this same graph.
- Give each group exactly ONE positional relation (inside, at_center_of,
  ring_around, cluster_near, along, on_shore_of or scatter_in). It may also
  have one orientation relation and any number of "avoid".
- Use "count" for things there are a specific number of -- one campfire, one
  well, six tents. Use "density" for things that fill space: trees, grass,
  rocks, rubble.
- If the description says "a" or "the" of something, that is ONE of it, and
  it is usually a focal point other things arrange around.
- Always include a region for the surroundings (wilderness) and, where there
  is a settlement or camp, a region for it.
- Include a water region only if the description mentions water. Give it an
  id that says which water it is: "river", "lake", or "sea" for a coast,
  harbour, beach or cove.
- Describe the place the prompt asks for, whatever it is -- a harbour, a
  snowbound pass, a swamp, a temple garden. Name assets plainly ("boat",
  "pier", "pine tree", "lantern"); if the library lacks one it is skipped and
  reported, so prefer common objects over exotic ones.
- Between 5 and 14 groups. Populate the place so it feels complete: describe
  what a prompt implies, not only what it names. A camp has tents, a fire,
  seating, stored gear, a woodpile, drying racks, and woodland around it.
- Scale the counts to the size of the scene you are told about. The same
  description on a large map means more of everything, not the same handful
  of objects lost in open ground.

EXAMPLE
Description: "a small forest camp around a campfire, beside a river"
{{
  "regions": [
    {{"id": "camp", "kind": "settlement", "size": "small"}},
    {{"id": "woods", "kind": "wilderness", "size": "large"}},
    {{"id": "river", "kind": "water", "size": "medium"}}
  ],
  "groups": [
    {{"id": "fire", "asset": "campfire", "count": 1,
      "relations": [{{"kind": "at_center_of", "target": "camp"}}]}},
    {{"id": "tents", "asset": "tent", "count": 5,
      "relations": [{{"kind": "ring_around", "target": "fire", "distance": 9}},
                    {{"kind": "faces", "target": "fire"}}]}},
    {{"id": "seats", "asset": "log", "count": 6,
      "relations": [{{"kind": "cluster_near", "target": "fire", "distance": 4}},
                    {{"kind": "faces", "target": "fire"}}]}},
    {{"id": "supplies", "asset": "crate", "count": 5,
      "relations": [{{"kind": "cluster_near", "target": "tents", "distance": 5}}]}},
    {{"id": "trees", "asset": "pine tree", "density": 1.8,
      "relations": [{{"kind": "scatter_in", "target": "woods"}},
                    {{"kind": "avoid", "target": "camp", "distance": 5}}]}},
    {{"id": "undergrowth", "asset": "bush", "density": 2.2,
      "relations": [{{"kind": "scatter_in", "target": "woods"}}]}},
    {{"id": "reeds", "asset": "grass", "density": 1.0,
      "relations": [{{"kind": "along", "target": "river", "distance": 4}}]}},
    {{"id": "stones", "asset": "rock", "density": 0.8,
      "relations": [{{"kind": "on_shore_of", "target": "river", "distance": 6}}]}}
  ]
}}
"""

GRAPH_USER = """Description:
"{prompt}"

The scene is {size} metres across, about {area} square metres of ground.
Populate it for that size: a place the player walks through, not a diorama.
As a guide, a settlement filling a scene this size wants roughly {buildings}
buildings or shelters, with the props and seating that go with them.

Return the scene graph."""


def _population_hint(size_metres: float) -> int:
    """Roughly how many buildings a settlement of this scale carries.

    Without a sense of scale the model writes the same four-tent camp whether
    the scene is sixty metres across or two hundred, and at two hundred that
    camp is lost in forty thousand square metres of forest.
    """
    if size_metres <= 70:
        return 5
    if size_metres <= 130:
        return 10
    return 18


def ask_for_graph(prompt: str, size_metres: float = 120.0,
                  client: Optional[LLMClient] = None) -> Optional[SceneGraph]:
    """Have the model compose a scene graph. None if it can't."""
    client = client or LLMClient()
    if not client.is_available():
        return None

    try:
        raw = client.generate_json(
            system=GRAPH_SYSTEM,
            user=GRAPH_USER.format(
                prompt=prompt,
                size=int(size_metres),
                area=f"{int(size_metres * size_metres):,}",
                buildings=_population_hint(size_metres)))
    except LLMUnavailable:
        return None

    try:
        raw.setdefault("regions", [])
        raw.setdefault("groups", [])
        graph = SceneGraph.model_validate(raw)
    except Exception as e:
        return None

    # A graph with nothing in it is worse than the fallback.
    if len(graph.groups) < 2:
        return None

    _ensure_surroundings(graph)
    return graph


def _ensure_surroundings(graph: SceneGraph) -> None:
    """Guarantee a region for groups that reference nothing.

    A group with no positional relation has to go somewhere, and "anywhere in
    the scene" is the sensible reading.
    """
    has_wilderness = any(r.kind == "wilderness" for r in graph.regions)
    orphans = [g for g in graph.groups if g.placement is None]

    if orphans and not has_wilderness:
        graph.regions.append(Region(id="surroundings", kind="wilderness",
                                    size="large"))
        has_wilderness = True

    if orphans:
        target = next(r.id for r in graph.regions if r.kind == "wilderness")
        for group in orphans:
            group.relations.insert(
                0, Relation(kind="scatter_in", target=target))
            graph.warnings.append(
                f"{group.id}: no position given; scattered in {target}")


# ---------------------------------------------------------------------------
# Route 2: compile a graph from the flat specification
# ---------------------------------------------------------------------------

# Which region each category belongs in, and how it sits there. These are the
# rules the graph exists to replace -- kept here, in one place, so the
# fallback still works when the model is unavailable.
_CATEGORY_PLACEMENT = {
    "building": ("settlement", "inside"),
    "structure": ("settlement", "inside"),
    "light_source": ("settlement", "inside"),
    "prop": ("settlement", "cluster_near"),
    "vegetation": ("wilderness", "scatter_in"),
    "rock": ("wilderness", "scatter_in"),
}

# Roughly how many of a thing fill 100 square metres, by category. Used to
# turn the flat spec's absolute counts into densities for scatter groups, so
# a bigger scene gets proportionally more.
_SCATTER_DENSITY = {"vegetation": 2.2, "rock": 1.0}


def compile_graph(spec) -> SceneGraph:
    """Derive a scene graph from the old flat specification."""
    regions: List[Region] = [
        Region(id="settlement", kind="settlement", size=spec.terrain.size),
        Region(id="wilderness", kind="wilderness", size="large"),
    ]

    for feature in getattr(spec, "features", []) or []:
        if feature.holds_water:
            regions.append(Region(id=feature.type, kind="water",
                                  size=feature.size, feature=feature.type))
        elif feature.type == "hill":
            regions.append(Region(id="high_ground", kind="high_ground",
                                  size=feature.size, feature="hill"))

    water_regions = [r.id for r in regions if r.kind == "water"]

    groups: List[Group] = []
    anchor_id: Optional[str] = None
    used_ids: set = set()

    def unique(base: str) -> str:
        slug = base.replace(" ", "_")[:24] or "group"
        candidate, n = slug, 2
        while candidate in used_ids:
            candidate, n = f"{slug}_{n}", n + 1
        used_ids.add(candidate)
        return candidate

    # A single light source is the scene's focal point -- a campfire, a
    # brazier. Everything sociable arranges around it.
    for obj in spec.objects:
        if obj.category == "light_source" and obj.quantity == 1:
            anchor_id = unique(obj.name)
            groups.append(Group(
                id=anchor_id, asset=obj.name, count=1,
                relations=[Relation(kind="at_center_of",
                                    target="settlement")]))
            break

    for obj in spec.objects:
        if anchor_id and obj.name.replace(" ", "_")[:24] == anchor_id:
            continue

        region, kind = _CATEGORY_PLACEMENT.get(
            obj.category, ("wilderness", "scatter_in"))
        gid = unique(obj.name)
        relations: List[Relation] = []

        if kind == "scatter_in":
            relations.append(Relation(kind="scatter_in", target=region))
            if obj.category == "vegetation":
                relations.append(Relation(kind="avoid", target="settlement",
                                          distance=6))
            density = _SCATTER_DENSITY.get(obj.category, 1.0)
            groups.append(Group(id=gid, asset=obj.name, density=density,
                                relations=relations))
            continue

        if obj.category == "building" and anchor_id:
            relations.append(Relation(kind="ring_around", target=anchor_id,
                                      distance=12))
            relations.append(Relation(kind="faces", target=anchor_id))
        elif kind == "cluster_near" and anchor_id:
            relations.append(Relation(kind="cluster_near", target=anchor_id,
                                      distance=8))
        else:
            relations.append(Relation(kind="inside", target=region))

        groups.append(Group(id=gid, asset=obj.name, count=obj.quantity,
                            relations=relations))

    # A coast is a working edge as well as scenery: boats drawn up on the
    # shore and a pier into the water say "harbour" more than any building.
    if any(r.feature == "coast" for r in regions):
        sea = next(r.id for r in regions if r.feature == "coast")
        groups.append(Group(
            id=unique("boats"), asset="boat", count=4,
            relations=[Relation(kind="on_shore_of", target=sea, distance=4)]))

    # Water earns a little life of its own, so a river is not just an
    # obstacle everything keeps away from.
    if water_regions:
        groups.append(Group(
            id=unique("reeds"), asset="grass", density=0.9,
            relations=[Relation(kind="along", target=water_regions[0],
                                distance=4)]))
        groups.append(Group(
            id=unique("shore stones"), asset="rock", density=0.6,
            relations=[Relation(kind="on_shore_of", target=water_regions[0],
                                distance=6)]))

    graph = SceneGraph(regions=regions, groups=groups)
    _ensure_surroundings(graph)
    return graph


def build_graph(prompt: str, spec, client: Optional[LLMClient] = None,
                force_fallback: bool = False,
                size_metres: float = 120.0) -> tuple:
    """Get a scene graph however we can. Returns (graph, source)."""
    if not force_fallback:
        graph = ask_for_graph(prompt, size_metres=size_metres, client=client)
        if graph is not None:
            return graph, "llm"
    return compile_graph(spec), "compiled"
