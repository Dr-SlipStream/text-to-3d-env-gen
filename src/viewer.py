"""
Stage 5b: a real-time viewer for the generated scene.

Writes a self-contained HTML file that loads the exported .glb and renders it
with atmosphere. This is where most of the perceived visual quality comes
from: the assets are stylised low-poly either way, but low-poly lit well reads
as deliberate art direction, while low-poly lit flatly reads as unfinished.

Three things here were wrong in the first version and are worth keeping
straight, because each one alone ruins the image:

  Fog density must scale with scene size. FogExp2 attenuates by
  exp(-(density * distance)^2), so a value tuned for a small scene turns a
  120m landscape into flat grey -- at density 0.0135, a point 100m away is
  84% fog. Densities here are expressed as a fraction of scene size and
  converted at write time.

  A terrain plane on its own is a floating island. With nothing behind it you
  see its unlit underside and a hard edge against the sky. A large ground
  plane at the terrain's base height, fading into fog, gives a horizon.

  The camera must start above the scene looking down into it. Starting far
  away and near ground level frames the terrain edge-on as a thin sliver.

three.js loads from a CDN, so there is nothing to install.
"""

from __future__ import annotations

import json
from pathlib import Path

# Sun colour, intensity and elevation per time of day -- the single biggest
# lever on how a scene reads.
LIGHTING = {
    "dawn":  {"sun": "#ffcf9c", "intensity": 2.4, "elevation": 14,
              "ambient": "#7c8db0", "ambient_i": 0.75,
              "sky_top": "#5b82b8", "sky_bottom": "#f0cfa4"},
    "day":   {"sun": "#fff4e2", "intensity": 3.1, "elevation": 55,
              "ambient": "#a9c0dc", "ambient_i": 0.85,
              "sky_top": "#5a8fce", "sky_bottom": "#d4e6f7"},
    "dusk":  {"sun": "#ffab6b", "intensity": 2.3, "elevation": 11,
              "ambient": "#6d7396", "ambient_i": 0.70,
              "sky_top": "#3d4670", "sky_bottom": "#d98a5c"},
    # Night has to stay readable. A physically plausible night render is
    # almost black, which is useless for showing what was generated -- games
    # solve this the same way, with a bright artificial "moon" and lifted
    # ambient rather than true darkness.
    #
    # The moon colour is kept close to neutral. Every light in a night scene
    # pulling towards blue turned green foliage teal and orange tents pink --
    # the scene reads as night from its darkness and sky, not from staining
    # every surface blue.
    "night": {"sun": "#dae2f2", "intensity": 1.35, "elevation": 42,
              "ambient": "#6d7794", "ambient_i": 0.95,
              "sky_top": "#1b2450", "sky_bottom": "#44507a"},
}

# Fog strength as a multiple of 1/scene_size, so a 60m scene and a 200m scene
# end up with the same *visual* amount of haze.
WEATHER_FOG = {
    "clear":     {"k": 0.22, "colour": None},
    "fog":       {"k": 0.75, "colour": "#b9c4cc"},
    "rain":      {"k": 0.48, "colour": "#8d97a3"},
    "storm":     {"k": 0.60, "colour": "#6b7280"},
    "snow":      {"k": 0.55, "colour": "#d8e2ea"},
    "sandstorm": {"k": 0.80, "colour": "#c9a878"},
}

MOOD_EXPOSURE = {
    "peaceful": 1.05, "lively": 1.15, "mysterious": 0.95,
    "tense": 0.92, "abandoned": 0.98,
}

# Night scenes need extra exposure on top of the mood adjustment, or a tense
# night ends up at 0.92 exposure on already-dim lighting and reads as black.
TIME_EXPOSURE = {"night": 1.35, "dusk": 1.1, "dawn": 1.1, "day": 1.0}


def _desaturate(hex_colour: str, towards_white: float = 0.55) -> str:
    """Pull a colour towards white.

    Sky colours are chosen to look right *as sky*, which means saturated. Used
    directly as the hemisphere light's upward colour they tint every surface
    in the scene: a strongly blue sky over green foliage renders it cyan, and
    that is exactly what happened -- trees came out turquoise under both the
    day and night skies, while the dusk scene, whose sky top is nearly grey,
    looked correct.
    """
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    t = towards_white
    r = int(r + (255 - r) * t)
    g = int(g + (255 - g) * t)
    b = int(b + (255 - b) * t)
    return f"#{r:02x}{g:02x}{b:02x}"

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { overflow:hidden; background:#0d1330;
         font-family:'Segoe UI',system-ui,sans-serif; }
  #c { display:block; width:100vw; height:100vh; }
  #info {
    position:fixed; top:16px; left:16px; padding:14px 18px;
    background:rgba(12,16,32,.68); backdrop-filter:blur(12px);
    border:1px solid rgba(255,255,255,.10); border-radius:10px;
    color:#e8ecf4; font-size:13px; line-height:1.65; max-width:330px;
    pointer-events:none;
  }
  #info h1 { font-size:14px; font-weight:600; margin-bottom:8px; color:#fff; }
  #info .p { color:#9aa6bd; font-style:italic; margin-bottom:10px;
             font-size:12px; line-height:1.5; }
  #info .r { display:flex; justify-content:space-between; gap:20px; }
  #info .k { color:#8b97ae; }
  #info .v { color:#dfe6f2; font-variant-numeric:tabular-nums; }
  #hint {
    position:fixed; bottom:16px; left:16px; padding:10px 14px;
    background:rgba(12,16,32,.68); backdrop-filter:blur(12px);
    border:1px solid rgba(255,255,255,.10); border-radius:8px;
    color:#9aa6bd; font-size:12px; line-height:1.6;
  }
  #hint b { color:#dfe6f2; font-weight:600; }
  #mode { color:#7dd3a0; }
  #load {
    position:fixed; inset:0; display:flex; align-items:center;
    justify-content:center; background:#0d1330; color:#8b97ae;
    font-size:14px; z-index:10; transition:opacity .5s;
  }
</style>
</head>
<body>
<div id="load">Loading scene…</div>
<canvas id="c"></canvas>
<div id="info">
  <h1>__TITLE__</h1>
  <div class="p">"__PROMPT__"</div>
  __ROWS__
</div>
<div id="hint">
  <b>Drag</b> orbit &nbsp; <b>Scroll</b> zoom &nbsp; <b>W</b> walk mode
  &nbsp; <b>R</b> reset &nbsp; <span id="mode"></span>
</div>

<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
} }
</script>

<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const CFG = __CONFIG__;
const S = CFG.size, MID = S / 2;

const renderer = new THREE.WebGLRenderer({
  canvas: document.getElementById('c'), antialias: true
});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = CFG.exposure;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const fogColour = new THREE.Color(CFG.fogColour);
scene.fog = new THREE.FogExp2(fogColour, CFG.fogDensity);

// --- sky -------------------------------------------------------------------
// The horizon band blends towards the fog colour so sky and fogged distance
// meet without a visible seam.
const skyGeo = new THREE.SphereGeometry(S * 6, 32, 16);
const skyMat = new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, fog: false,
  uniforms: {
    top: { value: new THREE.Color(CFG.skyTop) },
    bottom: { value: new THREE.Color(CFG.skyBottom) },
    haze: { value: fogColour }
  },
  vertexShader: `varying vec3 vP; void main(){ vP = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: `
    uniform vec3 top; uniform vec3 bottom; uniform vec3 haze;
    varying vec3 vP;
    void main(){
      float h = normalize(vP).y;
      vec3 sky = mix(bottom, top, pow(clamp(h, 0.0, 1.0), 0.55));
      float horizon = 1.0 - smoothstep(-0.02, 0.22, h);
      gl_FragColor = vec4(mix(sky, haze, horizon * 0.85), 1.0);
    }`
});
scene.add(new THREE.Mesh(skyGeo, skyMat));

// --- horizon ground --------------------------------------------------------
// Without this the terrain is a floating island: you see its unlit underside
// and a hard edge against the sky. A large fogged plane at the terrain's base
// height reads as land continuing to the horizon.
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(S * 9, 48),
  new THREE.MeshStandardMaterial({
    color: new THREE.Color(CFG.groundColour), roughness: 1.0, metalness: 0.0
  })
);
ground.rotation.x = -Math.PI / 2;
ground.position.set(MID, CFG.groundY, MID);
scene.add(ground);

// --- lighting --------------------------------------------------------------
// The hemisphere light uses a desaturated sky colour. Feeding it the sky's
// own colour tints every surface -- saturated blue over green foliage reads
// as cyan.
scene.add(new THREE.HemisphereLight(
  new THREE.Color(CFG.skyLight), new THREE.Color(CFG.ambient), CFG.ambientI));

const sun = new THREE.DirectionalLight(new THREE.Color(CFG.sun), CFG.sunI);
const el = CFG.elevation * Math.PI / 180, az = Math.PI * 0.35;
const d = S * 1.2;
sun.position.set(MID + Math.cos(el) * Math.cos(az) * d,
                 Math.sin(el) * d + 10,
                 MID + Math.cos(el) * Math.sin(az) * d);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 1;
sun.shadow.camera.far = S * 4;
const sh = S * 0.7;
sun.shadow.camera.left = -sh; sun.shadow.camera.right = sh;
sun.shadow.camera.top = sh;   sun.shadow.camera.bottom = -sh;
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.04;
sun.target.position.set(MID, 0, MID);
scene.add(sun, sun.target);

// Fill from the opposite side so shadowed faces don't crush to black. Night
// leans on this more heavily, since there is no sun to model.
const fill = new THREE.DirectionalLight(new THREE.Color(CFG.skyLight),
                                        CFG.fillI);
fill.position.set(MID - d * 0.6, d * 0.4, MID - d * 0.6);
scene.add(fill);

// --- camera ----------------------------------------------------------------
// Start above the scene looking down into it.
const camera = new THREE.PerspectiveCamera(
  50, innerWidth / innerHeight, 0.1, S * 20);
const home = new THREE.Vector3(MID - S * 0.42, S * 0.40, MID + S * 0.62);
camera.position.copy(home);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(MID, 0, MID);
controls.maxPolarAngle = Math.PI * 0.48;   // stay above the ground plane
controls.minDistance = 6;
controls.maxDistance = S * 2.2;

// --- scene load ------------------------------------------------------------
// Assets are loaded once each and copied to every instance transform, rather
// than baked into a single merged file. This keeps each pack's own textures
// and materials intact -- flattening them to one colour per model is what
// turned bushes cyan and walls navy.
const loader = new GLTFLoader();
const loadingEl = document.getElementById('load');

function tuneMaterials(root) {
  root.traverse((o) => {
    if (!o.isMesh) return;
    o.castShadow = true;
    o.receiveShadow = true;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of mats) {
      if (!m) continue;
      // Packs ship shiny defaults; roughening reads as painted stylised art
      // rather than plastic. We nudge rather than overwrite, so texture and
      // colour data survives.
      if (m.roughness !== undefined) m.roughness = Math.max(m.roughness, 0.7);
      if (m.metalness !== undefined) m.metalness = Math.min(m.metalness, 0.05);
      m.flatShading = false;
      if (m.map) m.map.colorSpace = THREE.SRGBColorSpace;
      m.needsUpdate = true;
    }
  });
}

function load(url) {
  return new Promise((res, rej) => loader.load(url, res, undefined, rej));
}

async function build() {
  const manifest = await fetch(CFG.manifest).then(r => r.json());

  if (manifest.terrain) {
    const g = await load(manifest.terrain);
    tuneMaterials(g.scene);
    g.scene.traverse((o) => { if (o.isMesh) o.castShadow = false; });
    scene.add(g.scene);
  }

  // Load every distinct model once.
  const models = new Map();
  await Promise.all(manifest.assets.map(async (a) => {
    try {
      const g = await load(a.file);
      tuneMaterials(g.scene);
      models.set(a.file, { root: g.scene, offset: a.base_offset || 0 });
    } catch (e) {
      console.warn('could not load', a.file, e);
    }
  }));

  // Place the copies.
  const group = new THREE.Group();
  const lightSpots = [];
  let placed = 0;
  for (const inst of manifest.instances) {
    const model = models.get(inst.asset);
    if (!model) continue;

    const obj = model.root.clone(true);
    obj.position.set(inst.p[0], inst.p[1] - model.offset * inst.s, inst.p[2]);
    obj.rotation.y = inst.r;
    obj.scale.setScalar(inst.s);
    group.add(obj);
    placed++;

    if (inst.category === 'light_source') lightSpots.push(inst.p);
  }
  scene.add(group);

  // Torches and campfires were being placed as geometry that emitted no
  // light, so a night scene had lanterns everywhere and no glow. Real point
  // lights are added at a subset of them -- a subset because hundreds of
  // dynamic lights would not render at an interactive frame rate.
  if (CFG.lightsEmit && lightSpots.length) {
    const budget = Math.min(lightSpots.length, CFG.maxPointLights);
    const step = Math.max(1, Math.floor(lightSpots.length / budget));
    for (let i = 0; i < lightSpots.length && group.children.length; i += step) {
      const p = lightSpots[i];
      const lamp = new THREE.PointLight(
        new THREE.Color(CFG.lightColour), CFG.lightIntensity, CFG.lightRange, 2);
      lamp.position.set(p[0], p[1] + 1.2, p[2]);
      scene.add(lamp);
      // A small emissive ball so the source itself reads as lit, not just
      // the ground around it.
      const bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.22, 8, 6),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(CFG.lightColour) })
      );
      bulb.position.copy(lamp.position);
      scene.add(bulb);
    }
  }

  console.log(`placed ${placed} instances from ${models.size} models`);
  loadingEl.style.opacity = '0';
  setTimeout(() => loadingEl.remove(), 500);
}

build().catch((err) => {
  loadingEl.textContent =
    'Could not load the scene — serve this folder over HTTP.';
  console.error(err);
});

// --- controls --------------------------------------------------------------
let walk = false, yaw = Math.PI, pitch = -0.05;
let dragging = false, lx = 0, ly = 0;
const keys = {};
const modeEl = document.getElementById('mode');

function setWalk(on) {
  walk = on;
  controls.enabled = !on;
  modeEl.textContent = on ? 'WALK — arrows or A/D to move, drag to look' : '';
  if (on) {
    camera.position.set(MID, CFG.terrainMax + CFG.eyeHeight, MID + S * 0.42);
    yaw = Math.PI; pitch = -0.05;
  } else {
    camera.position.copy(home);
    controls.target.set(MID, 0, MID);
  }
}

addEventListener('keydown', (e) => { keys[e.code] = true; });
addEventListener('keyup', (e) => { keys[e.code] = false; });
addEventListener('keypress', (e) => {
  const k = e.key.toLowerCase();
  if (k === 'w') setWalk(!walk);
  if (k === 'r') setWalk(false);
});

renderer.domElement.addEventListener('mousedown', (e) => {
  if (walk) { dragging = true; lx = e.clientX; ly = e.clientY; }
});
addEventListener('mouseup', () => { dragging = false; });
addEventListener('mousemove', (e) => {
  if (!walk || !dragging) return;
  yaw -= (e.clientX - lx) * 0.004;
  pitch = Math.max(-1.2, Math.min(1.0, pitch - (e.clientY - ly) * 0.004));
  lx = e.clientX; ly = e.clientY;
});

const clock = new THREE.Clock();
function tick() {
  requestAnimationFrame(tick);
  const dt = Math.min(clock.getDelta(), 0.1);

  if (walk) {
    const speed = (keys.ShiftLeft ? 26 : 10) * dt;
    const fwd = new THREE.Vector3(
      Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch),
      Math.cos(yaw) * Math.cos(pitch));
    const flat = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
    const right = new THREE.Vector3(
      Math.sin(yaw - Math.PI / 2), 0, Math.cos(yaw - Math.PI / 2));
    if (keys.ArrowUp) camera.position.addScaledVector(flat, speed);
    if (keys.ArrowDown) camera.position.addScaledVector(flat, -speed);
    if (keys.ArrowLeft || keys.KeyA) camera.position.addScaledVector(right, speed);
    if (keys.ArrowRight || keys.KeyD) camera.position.addScaledVector(right, -speed);
    camera.lookAt(camera.position.clone().add(fwd));
  } else {
    controls.update();
  }
  renderer.render(scene, camera);
}
tick();

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>
"""


def write_viewer(scene, out_dir: Path,
                 manifest_file: str = "scene_manifest.json",
                 title: str = "Generated Environment") -> Path:
    """Write index.html next to the exported .glb."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = scene.spec
    tod = spec.lighting.time_of_day if spec else "day"
    weather = spec.lighting.weather if spec else "clear"
    mood = spec.lighting.mood if spec else "peaceful"

    light = LIGHTING.get(tod, LIGHTING["day"])
    fog = WEATHER_FOG.get(weather, WEATHER_FOG["clear"])

    terrain = scene.terrain
    size = terrain.size if terrain is not None else 120.0
    ground_y = float(terrain.heights.min()) if terrain is not None else 0.0
    terrain_max = float(terrain.heights.max()) if terrain is not None else 2.0

    # Colour the horizon plane like the terrain's low ground, so the generated
    # patch blends into surrounding land rather than sitting on top of it.
    if terrain is not None:
        # Match the terrain's own low colour. Darkening it produced a
        # visible ring where the generated patch met the surrounding land,
        # which made the scene read as a floating tile.
        lo = terrain.preset.colour_low
        hi = terrain.preset.colour_high
        c = tuple((lo[i] * 0.65 + hi[i] * 0.35) for i in range(3))
        ground_colour = "#%02x%02x%02x" % tuple(
            max(0, min(255, int(v * 255))) for v in c)
    else:
        ground_colour = "#3a4a2a"

    fog_colour = fog["colour"] or light["sky_bottom"]
    # Relative to scene size, so haze looks the same at any scale.
    density = fog["k"] / size
    if tod == "night":
        density *= 1.15

    config = {
        "manifest": manifest_file,
        "size": size,
        "groundY": round(ground_y - 0.15, 3),
        "terrainMax": round(terrain_max, 3),
        "groundColour": ground_colour,
        "eyeHeight": 1.7,
        "sun": _desaturate(light["sun"], 0.25),
        "sunI": light["intensity"],
        "elevation": light["elevation"],
        # Desaturated for the same reason as the sky colour: this is the
        # hemisphere light's *ground* colour and tints everything lit from
        # below.
        "ambient": _desaturate(light["ambient"], 0.45),
        "ambientI": light["ambient_i"],
        "fillI": 0.75 if tod == "night" else 0.45,
        # Only worth the cost when it's dark enough to see them.
        "lightsEmit": tod in ("night", "dusk"),
        "maxPointLights": 14 if tod == "night" else 8,
        "lightColour": "#ffb265",
        "lightIntensity": 45.0 if tod == "night" else 22.0,
        "lightRange": 22.0,
        "skyTop": light["sky_top"],
        "skyLight": _desaturate(light["sky_top"]),
        "skyBottom": light["sky_bottom"],
        "fogColour": fog_colour,
        "fogDensity": round(density, 6),
        "exposure": round(MOOD_EXPOSURE.get(mood, 1.0)
                          * TIME_EXPOSURE.get(tod, 1.0), 3),
    }

    theme_label = spec.theme.replace("_", " ") if spec else "-"
    if spec is not None and not getattr(spec, "theme_recognised", True):
        # Say so on the scene itself. A viewer that quietly shows a village
        # for "an underwater city" is misleading; one that labels the
        # substitution is honest about the system's range.
        theme_label += "  (approximated)"

    rows = [
        ("Theme", theme_label),
        ("Terrain", f"{spec.terrain.type.replace('_', ' ')} · {int(size)} m"
                    if spec else "-"),
        ("Lighting", f"{tod} · {weather} · {mood}"),
        ("Objects", f"{len(scene.instances)} instances"),
        ("Triangles", f"{scene.total_triangles:,}"),
    ]
    rows_html = "\n  ".join(
        f'<div class="r"><span class="k">{k}</span>'
        f'<span class="v">{v}</span></div>' for k, v in rows)

    prompt = (spec.source_prompt if spec and spec.source_prompt
              else title).replace('"', "&quot;")

    html = (HTML
            .replace("__CONFIG__", json.dumps(config))
            .replace("__ROWS__", rows_html)
            .replace("__PROMPT__", prompt)
            .replace("__TITLE__", title))

    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
