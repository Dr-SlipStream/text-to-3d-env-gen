# Unity Import

How to get a generated scene into Unity, and what to say about it.

The whole point of exporting glTF rather than to a Unity-specific format is
that the pipeline stays engine-agnostic: the same file opens in Unity, Unreal,
Blender, or the browser. This guide covers the Unity half.

---

## One-time setup

### 1. Install Unity

Unity Hub, then any Unity **6** or **2022 LTS** editor. Anything from
2021.3.46f1 onwards works. A 3D (URP) template is fine; so is Built-In.

### 2. Install glTFast — do this before anything else

Unity does not read `.glb` out of the box. Its official package does.

**Install it before you try to import anything.** Without it Unity does not
recognise `.glb` as an importable type at all, and refuses the drop with a
"no entry" cursor (a circle with a slash). That looks like a permissions or
path problem and isn't one -- Unity simply doesn't know what the file is.

If you create a fresh project for the demo, this step is easy to forget.

1. **Window → Package Manager**
2. **+** button → **Add package by name**
3. Enter `com.unity.cloud.gltfast`
4. **Add**, leave the version field empty

That package registers itself as the **default importer for `.gltf` and
`.glb`**, so from then on a GLB dropped into `Assets/` imports automatically,
the same way an FBX would. It's Apache-licensed, made by Unity, and supports
Built-In, URP and HDRP.

> If Unity reports *"Multiple scripted importers are targeting the extension
> 'glb'"*, another glTF plugin is already installed. Remove it, or demote it
> to an alternative importer.

---

## Generating a Unity-ready scene

```powershell
python scripts/generate.py "a medieval village on a sunny day" --size large --seed 7 --unity
```

`--unity` adds a `unity/` subfolder to the output:

```
outputs/a-medieval-village-on-a-sunny-day/
    scene_manifest.json        every instance's transform
    terrain.glb                the ground
    assets/                    each distinct model, stored once
    index.html                 the browser viewer
    unity/
        scene.glb              the whole scene as one file
        Editor/
            GeneratedSceneImporter.cs
        README.txt
```

---

## Route A — generate straight into the project

**Use this for the demo.** It removes the drag entirely, and with it the
commonest failure.

```powershell
python scripts/generate.py "a medieval village on a sunny day" `
    --size large --seed 7 --palette `
    --unity-project "C:\Users\rohit\My project"
```

Point `--unity-project` at the folder containing `Assets`, `Packages` and
`ProjectSettings`. The baked `scene.glb` is written into
`Assets/GeneratedScenes/<name>/`.

Only that one file is copied. The per-model assets are copies of the original
pack files, and a handful of those fail Unity's glTF importer for reasons of
their own -- copying them in fills the console with errors about models the
drag-and-drop route never uses. Add `--unity-full` when you specifically want
the manifest route (Route B), which needs them.

Switch to Unity. It notices the new files and imports them on its own.

Then drag the imported `scene.glb` from the **Project** window into the
**Hierarchy** and the environment appears.

### Why not just drag the file in from Explorer?

Unity can only import files that physically live under `Assets/`. Dragging a
`.glb` from anywhere else — or dropping it onto the Inspector rather than the
Project window — fails with:

```
Invalid AssetDatabase path: C:/.../unity/scene.glb.
Use path relative to the project folder.
```

If you do want to do it by hand: copy `scene.glb` into `Assets/` using
**Windows Explorer**, then switch back to Unity. Copying into the folder always
works; dragging across the window boundary is what's fragile.

The whole environment appears — terrain, buildings, trees, props — in one
prefab, correctly positioned.

**Say this while it imports:**

> "This is a standard glTF file. Unity's own importer reads it directly — no
> conversion, no plugin of ours, no manual fixing. The same file opens in
> Unreal or Blender."

---

## Route B — engine-native, rebuilt from the manifest

More impressive, and closer to how a level is actually authored: one mesh in
memory, referenced by many objects, rather than thousands of duplicated
meshes.

1. Copy the **entire scene folder** into `Assets/GeneratedScenes/`.
   Unity imports every `.glb` inside it automatically.
2. **Tools → Generated Scene → Import from manifest...**
3. Browse to that folder's `scene_manifest.json`.
4. **Build Scene**.

The importer reads each instance's position, rotation and scale, instantiates
the shared model at each one, groups them by category so the hierarchy is
navigable, adds mesh colliders, and marks everything static so Unity can batch
it and bake lighting.

**Options in the window**

| Option | What it does |
|---|---|
| Convert handedness | glTF is right-handed, Unity is left-handed. glTFast mirrors X when importing each model, so instance positions are mirrored to match. Turn off only if the layout comes out reversed. |
| Add mesh colliders | Makes the scene walkable straight away. Vegetation is skipped — walking through undergrowth is expected. |
| Mark static | Enables batching, occlusion and baked GI. Without it, thousands of small objects each cost a draw call. |

**Say this:**

> "The manifest lists every object's transform, and each model is stored once.
> This rebuilds the scene the way an engine expects it — shared meshes,
> grouped hierarchy, colliders, static batching. That's the difference between
> exporting a picture of a level and exporting a level."

---

## Lighting the imported scene

**glTF carries geometry and materials. It has no concept of fog, ambient light
or exposure**, and engines light scenes with their own systems. So an imported
scene arrives lit by whatever that Unity project's default directional light
happens to be -- which is why a night scene can come in looking like an
overcast afternoon, with campfires that are just unlit props.

The generator writes `lighting.json` alongside the model. To reproduce the
browser preview:

**Tools → Generated Scene → Apply Lighting...** → pick `lighting.json`

That sets:

| Setting | From |
|---|---|
| Directional light colour, intensity, angle | time of day |
| Ambient sky / equator / ground colours | desaturated sky and ground tints |
| Fog colour and exponential-squared density | weather, scaled to scene size |
| A warm point light at each torch and campfire | the light-source positions in the scene |

The lamp count is capped, because hundreds of dynamic lights will not render
at an interactive frame rate.

**One thing the script cannot set: exposure.** In Unity that is a
post-processing setting, not a scene setting. For a URP project, add a
**Tonemapping** and **Exposure** override to the Global Volume and use the
value the script reports.

Both the browser viewer and this script read the same `lighting_config()`
function on the Python side, so they cannot drift apart. There is a test
asserting they agree.

### What to say

> "glTF is a geometry format — it deliberately doesn't carry lighting, because
> every engine does lighting differently. We export the scene's lighting as
> data alongside it, and a small editor script applies it. That's the same
> data the browser preview uses, so both show the same thing."

---

## Walking around it

1. **GameObject → 3D Object → Capsule**, place it above the terrain.
2. Add a **Character Controller** component.
3. Drop in any first-person controller script, or use Unity's Starter Assets.

Because the importer adds mesh colliders, the ground and buildings are solid
immediately.

---

## What makes the import clean

These are deliberate choices, and worth naming if asked:

| Property | Why it matters |
|---|---|
| **glTF 2.0, no required extensions** | Draco, KTX and meshopt compression each need an extra Unity package. Requiring one turns a drag-and-drop into a support call. We require none. |
| **Explicit PBR material** | trimesh writes vertex colours but no material, and the glTF *default* material is metallic 1.0 — which renders as dark metal in any engine without an environment map. We write an explicit non-metallic material so the file looks right everywhere, not just in our own viewer. |
| **Lighting exported as data** | glTF cannot carry fog, ambient or exposure. Rather than pretend, we export them alongside and provide a script that applies them. |
| **Vertex normals present** | Without them the engine falls back to flat per-face shading and every curved surface shows facets. |
| **Metres, Y-up** | Matches Unity's units and axis convention, so nothing needs rescaling. |
| **Relative asset paths** | The manifest never contains a path from the machine that generated it. |
| **Valid container lengths** | A declared size that disagrees with the file size is the classic reason an importer rejects a file outright. There's a test asserting this. |

Seven automated tests cover exactly these properties (`tests/test_unity_export.py`),
so "it imports cleanly" is checked on every run rather than hoped for.

---

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| **Drag is refused, "no entry" cursor** | **glTFast isn't installed.** Unity doesn't recognise `.glb` as importable, so it rejects the drop. Package Manager → + → Add package by name → `com.unity.cloud.gltfast`. This is by far the commonest cause, especially in a freshly created project. |
| "Invalid AssetDatabase path ... use path relative to the project folder" | The file is outside the project. Use `--unity-project`, or copy it into `Assets/` with Explorer. |
| `.glb` imports but shows as a plain file with no preview | glTFast installed but not yet compiled. Wait for Unity to finish, or right-click the file → Reimport. |
| "Failed to import ... assets/<something>.glb" | Individual source models that Unity's importer rejects. They only matter for Route B. Delete the `assets/` folder from that scene in Unity — the baked `scene.glb` doesn't reference it and will keep working. Generate without `--unity-full` to avoid copying them at all. |
| "Failed to import ... assets/box.glb" | A scene generated while the test suite's temporary manifest was in place. Rebuild with `python scripts/ingest_assets.py`, delete that scene folder from Unity, regenerate. |
| Warning UAC1009: field uses an unsupported type `float[][]` | An older copy of `SceneLightingApplier.cs`. Harmless — the field was never used — but replace the script with the current version to clear it. |

## Housekeeping

Each generated scene gets its own folder under `Assets/GeneratedScenes/`. They
are self-contained, so old ones can simply be deleted from the Project window
when you no longer want them. Deleting a scene folder removes its models too,
since nothing else references them.
| "Multiple scripted importers are targeting 'glb'" | Another glTF plugin is installed. Remove it. |
| Everything is dark grey or black | Old export without the material fix. Regenerate. |
| Scene appears mirrored | Toggle **Convert handedness** in the importer window. |
| Menu item missing | `GeneratedSceneImporter.cs` must be inside a folder named `Editor`. |
| Import is slow | Route B on a 2,000-object scene takes a few seconds. Route A is instant. |
| Objects float or sink | Regenerate — older exports predate the base-offset fix. |

---

## The point to land

> "We export glTF, the standard interchange format, rather than to any one
> engine. That means the same output opens in Unity, Unreal, Blender or a
> browser — and it means the pipeline isn't tied to a vendor. The Unity import
> is a drag-and-drop because the file is spec-clean, not because we wrote a
> Unity plugin."
