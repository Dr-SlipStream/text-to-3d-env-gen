// SceneLightingApplier.cs
//
// Reproduces a generated scene's lighting inside Unity.
//
// glTF carries geometry and materials, but has no concept of fog, ambient
// light or exposure -- so an imported scene arrives lit by whatever the
// project's defaults happen to be, which is why a night scene can come in
// looking like an overcast afternoon.
//
// The generator writes lighting.json alongside the model. This reads it and
// configures the sun, ambient, fog, and a point light at each torch or
// campfire, so Unity shows what the browser viewer shows.
//
// Usage:
//   Tools > Generated Scene > Apply Lighting...
//
// Put this file anywhere under an Editor/ folder.

using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace TextTo3D
{
    [Serializable]
    public class SceneLighting
    {
        public string time_of_day;
        public string weather;
        public string mood;
        public float scene_size;

        public string sun_colour;
        public float sun_intensity;
        public float sun_elevation_deg;
        public float sun_azimuth_deg;

        public string ambient_sky;
        public string ambient_ground;
        public float ambient_intensity;

        public string fog_colour;
        public float fog_density;
        public float exposure;

        public bool lamps_emit;
        public string lamp_colour;
        public float lamp_intensity;
        public float lamp_range;
        public int max_lamps;

        // lamp_positions is deliberately absent. It is a nested array, which
        // JsonUtility cannot deserialise, and declaring it as float[][] makes
        // Unity's serialization analyzer warn (UAC1009) about a field it will
        // never populate. ParseLampPositions() reads it out of the raw JSON
        // instead.
    }

    public class SceneLightingApplier : EditorWindow
    {
        string _path = "";
        bool _mirrorX = true;
        bool _placeLamps = true;
        string _status = "";

        [MenuItem("Tools/Generated Scene/Apply Lighting...")]
        static void Open()
        {
            var w = GetWindow<SceneLightingApplier>(true, "Apply Scene Lighting");
            w.minSize = new Vector2(440, 240);
        }

        void OnGUI()
        {
            EditorGUILayout.LabelField("Scene Lighting", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Reads lighting.json from a generated scene and configures the "
                + "sun, ambient light, fog and campfire lights to match the "
                + "browser preview.", MessageType.Info);

            EditorGUILayout.Space();
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("lighting.json", GUILayout.Width(90));
            EditorGUILayout.SelectableLabel(
                string.IsNullOrEmpty(_path) ? "(none)" : _path,
                EditorStyles.textField,
                GUILayout.Height(EditorGUIUtility.singleLineHeight));
            if (GUILayout.Button("Browse", GUILayout.Width(70)))
            {
                var picked = EditorUtility.OpenFilePanel(
                    "Select lighting.json", Application.dataPath, "json");
                if (!string.IsNullOrEmpty(picked)) _path = picked;
            }
            EditorGUILayout.EndHorizontal();

            _mirrorX = EditorGUILayout.Toggle(
                new GUIContent("Convert handedness",
                    "Match the axis flip the model importer applies."),
                _mirrorX);
            _placeLamps = EditorGUILayout.Toggle(
                new GUIContent("Place campfire lights",
                    "Adds a warm point light at each torch or campfire. "
                    + "Capped, because hundreds of dynamic lights will not "
                    + "render at an interactive frame rate."),
                _placeLamps);

            EditorGUILayout.Space();
            GUI.enabled = !string.IsNullOrEmpty(_path);
            if (GUILayout.Button("Apply", GUILayout.Height(30))) Apply();
            GUI.enabled = true;

            if (!string.IsNullOrEmpty(_status))
            {
                EditorGUILayout.Space();
                EditorGUILayout.HelpBox(_status, MessageType.None);
            }
        }

        void Apply()
        {
            _status = "";
            if (!File.Exists(_path)) { _status = "File not found."; return; }

            SceneLighting cfg;
            string raw;
            try
            {
                raw = File.ReadAllText(_path);
                cfg = JsonUtility.FromJson<SceneLighting>(raw);
            }
            catch (Exception e)
            {
                _status = "Could not read: " + e.Message;
                return;
            }
            if (cfg == null) { _status = "Empty or invalid file."; return; }

            // ---- sun ----------------------------------------------------
            var sun = FindDirectionalLight();
            if (sun == null)
            {
                var go = new GameObject("Directional Light");
                sun = go.AddComponent<Light>();
                sun.type = LightType.Directional;
                Undo.RegisterCreatedObjectUndo(go, "Create sun");
            }
            Undo.RecordObject(sun, "Apply scene lighting");
            Undo.RecordObject(sun.transform, "Apply scene lighting");

            sun.color = ParseColour(cfg.sun_colour, Color.white);
            sun.intensity = cfg.sun_intensity;
            sun.shadows = LightShadows.Soft;
            // Elevation above the horizon, azimuth around it.
            sun.transform.rotation = Quaternion.Euler(
                cfg.sun_elevation_deg, cfg.sun_azimuth_deg, 0f);

            // ---- ambient ------------------------------------------------
            RenderSettings.ambientMode =
                UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor =
                ParseColour(cfg.ambient_sky, Color.grey) * cfg.ambient_intensity;
            RenderSettings.ambientEquatorColor =
                ParseColour(cfg.ambient_sky, Color.grey) * cfg.ambient_intensity
                * 0.8f;
            RenderSettings.ambientGroundColor =
                ParseColour(cfg.ambient_ground, Color.grey)
                * cfg.ambient_intensity;

            // ---- fog ----------------------------------------------------
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogColor = ParseColour(cfg.fog_colour, Color.grey);
            RenderSettings.fogDensity = cfg.fog_density;

            // ---- campfire lights ----------------------------------------
            int lamps = 0;
            var holderName = "GeneratedLights";
            var existing = GameObject.Find(holderName);
            if (existing != null) Undo.DestroyObjectImmediate(existing);

            if (_placeLamps && cfg.lamps_emit)
            {
                // JsonUtility cannot deserialise nested arrays, so the lamp
                // positions are pulled out of the raw JSON by hand.
                var positions = ParseLampPositions(raw);
                if (positions.Length > 0)
                {
                    var holder = new GameObject(holderName);
                    Undo.RegisterCreatedObjectUndo(holder, "Add lights");

                    var colour = ParseColour(cfg.lamp_colour,
                                             new Color(1f, 0.7f, 0.4f));
                    var budget = Mathf.Min(positions.Length,
                                           Mathf.Max(cfg.max_lamps, 1));
                    var step = Mathf.Max(1, positions.Length / budget);
                    var mirror = _mirrorX ? -1f : 1f;

                    for (int i = 0; i < positions.Length && lamps < budget;
                         i += step)
                    {
                        var p = positions[i];
                        var go = new GameObject("Firelight");
                        go.transform.SetParent(holder.transform);
                        go.transform.position = new Vector3(
                            p[0] * mirror, p[1] + 1.2f, p[2]);

                        var lamp = go.AddComponent<Light>();
                        lamp.type = LightType.Point;
                        lamp.color = colour;
                        lamp.intensity = cfg.lamp_intensity / 20f; // Unity units
                        lamp.range = cfg.lamp_range;
                        lamp.shadows = LightShadows.None;
                        lamps++;
                    }
                }
            }

            _status =
                $"Applied {cfg.time_of_day} / {cfg.weather} / {cfg.mood}.\n"
                + $"Sun {cfg.sun_colour} at {cfg.sun_intensity}, "
                + $"fog density {cfg.fog_density}.\n"
                + (lamps > 0 ? $"Placed {lamps} firelights."
                             : "No firelights (daylight scene).")
                + "\n\nNote: exposure is a post-processing setting. For a URP "
                + "project, set it on the Global Volume's Tonemapping/Exposure "
                + $"override -- the viewer uses {cfg.exposure}.";
        }

        static Light FindDirectionalLight()
        {
            foreach (var light in UnityEngine.Object.FindObjectsByType<Light>(
                         FindObjectsSortMode.None))
                if (light.type == LightType.Directional) return light;
            return null;
        }

        static Color ParseColour(string hex, Color fallback)
        {
            if (string.IsNullOrEmpty(hex)) return fallback;
            return ColorUtility.TryParseHtmlString(hex, out var c)
                ? c : fallback;
        }

        // "lamp_positions": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        static float[][] ParseLampPositions(string json)
        {
            var key = "\"lamp_positions\"";
            var at = json.IndexOf(key, StringComparison.Ordinal);
            if (at < 0) return Array.Empty<float[]>();

            var open = json.IndexOf('[', at);
            if (open < 0) return Array.Empty<float[]>();

            int depth = 0, close = open;
            for (int i = open; i < json.Length; i++)
            {
                if (json[i] == '[') depth++;
                else if (json[i] == ']')
                {
                    depth--;
                    if (depth == 0) { close = i; break; }
                }
            }

            var body = json.Substring(open + 1, close - open - 1);
            var parts = body.Split(new[] { '[' },
                                   StringSplitOptions.RemoveEmptyEntries);
            var result = new System.Collections.Generic.List<float[]>();

            foreach (var part in parts)
            {
                var end = part.IndexOf(']');
                if (end < 0) continue;
                var nums = part.Substring(0, end).Split(',');
                if (nums.Length < 3) continue;

                var vec = new float[3];
                var ok = true;
                for (int i = 0; i < 3; i++)
                    ok &= float.TryParse(nums[i].Trim(),
                        System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture,
                        out vec[i]);
                if (ok) result.Add(vec);
            }
            return result.ToArray();
        }
    }
}
