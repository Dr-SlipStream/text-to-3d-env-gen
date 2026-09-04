// GeneratedSceneImporter.cs
//
// Rebuilds a generated scene inside Unity from its manifest.
//
// The pipeline exports two ways:
//
//   --baked      one scene.glb containing everything. Drag it into the
//                project and it appears. Nothing in this file is needed.
//
//   instanced    terrain.glb, an assets/ folder holding each distinct model
//                once, and scene_manifest.json listing every instance's
//                position, rotation and scale. That is how a game engine
//                actually stores a level -- one mesh referenced many times
//                rather than thousands of copies -- and this script
//                reconstructs it as a real Unity hierarchy.
//
// Usage:
//   1. Install com.unity.cloud.gltfast via Package Manager (Add package by
//      name). It registers as the default importer for .glb, so every model
//      in the folder imports automatically.
//   2. Copy the whole output folder into Assets/GeneratedScenes/.
//   3. Tools > Generated Scene > Import from manifest...
//
// Put this file anywhere under an Editor/ folder.

using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace TextTo3D
{
    // ---- manifest schema, mirroring the Python side --------------------
    [Serializable]
    public class SceneAsset
    {
        public string file;
        public string name;
        public float base_offset;
        public int triangles;
        public int count;
    }

    [Serializable]
    public class SceneInstance
    {
        public string asset;
        public string name;
        public string category;
        public float[] p;       // position: x, y, z
        public float r;         // rotation about Y, radians
        public float s;         // uniform scale
    }

    [Serializable]
    public class SceneStats
    {
        public int instance_count;
        public int unique_assets;
        public int terrain_triangles;
        public int rendered_triangles;
    }

    [Serializable]
    public class SceneManifest
    {
        public int version;
        public string terrain;
        public float size;
        public float ground_y;
        public SceneAsset[] assets;
        public SceneInstance[] instances;
        public SceneStats stats;
    }

    public class GeneratedSceneImporter : EditorWindow
    {
        string _manifestPath = "";
        bool _mirrorX = true;
        bool _addColliders = true;
        bool _staticBatching = true;
        Vector2 _scroll;
        string _status = "";

        [MenuItem("Tools/Generated Scene/Import from manifest...")]
        static void Open()
        {
            var window = GetWindow<GeneratedSceneImporter>(true,
                "Import Generated Scene");
            window.minSize = new Vector2(460, 320);
        }

        void OnGUI()
        {
            _scroll = EditorGUILayout.BeginScrollView(_scroll);

            EditorGUILayout.LabelField("Generated Scene Importer",
                EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Copy the generated output folder into Assets/ first, so Unity "
                + "imports the .glb models. Then pick its scene_manifest.json.",
                MessageType.Info);

            EditorGUILayout.Space();
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Manifest", GUILayout.Width(70));
            EditorGUILayout.SelectableLabel(
                string.IsNullOrEmpty(_manifestPath) ? "(none selected)"
                                                    : _manifestPath,
                EditorStyles.textField,
                GUILayout.Height(EditorGUIUtility.singleLineHeight));
            if (GUILayout.Button("Browse", GUILayout.Width(70)))
            {
                var picked = EditorUtility.OpenFilePanel(
                    "Select scene_manifest.json", Application.dataPath, "json");
                if (!string.IsNullOrEmpty(picked)) _manifestPath = picked;
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Options", EditorStyles.boldLabel);

            _mirrorX = EditorGUILayout.Toggle(
                new GUIContent("Convert handedness",
                    "glTF is right-handed, Unity is left-handed. glTFast "
                    + "mirrors the X axis when it imports each model, so "
                    + "instance positions must be mirrored to match. Turn "
                    + "this off only if the layout comes out reversed."),
                _mirrorX);

            _addColliders = EditorGUILayout.Toggle(
                new GUIContent("Add mesh colliders",
                    "Makes the scene immediately walkable with a character "
                    + "controller."),
                _addColliders);

            _staticBatching = EditorGUILayout.Toggle(
                new GUIContent("Mark static",
                    "Flags instances as static so Unity can batch them and "
                    + "bake lighting. Thousands of small objects otherwise "
                    + "cost one draw call each."),
                _staticBatching);

            EditorGUILayout.Space();
            GUI.enabled = !string.IsNullOrEmpty(_manifestPath);
            if (GUILayout.Button("Build Scene", GUILayout.Height(32)))
                Build();
            GUI.enabled = true;

            if (!string.IsNullOrEmpty(_status))
            {
                EditorGUILayout.Space();
                EditorGUILayout.HelpBox(_status, MessageType.None);
            }

            EditorGUILayout.EndScrollView();
        }

        void Build()
        {
            _status = "";

            if (!File.Exists(_manifestPath))
            {
                _status = "Manifest not found.";
                return;
            }

            SceneManifest manifest;
            try
            {
                manifest = JsonUtility.FromJson<SceneManifest>(
                    File.ReadAllText(_manifestPath));
            }
            catch (Exception e)
            {
                _status = "Could not read manifest: " + e.Message;
                return;
            }

            if (manifest?.instances == null || manifest.instances.Length == 0)
            {
                _status = "Manifest contains no instances.";
                return;
            }

            // The manifest sits alongside terrain.glb and assets/, and paths
            // inside it are relative to that folder.
            var folder = Path.GetDirectoryName(_manifestPath);
            var assetsFolder = ToProjectRelative(folder);
            if (assetsFolder == null)
            {
                _status = "The output folder must live inside Assets/ so "
                          + "Unity can import the .glb models.";
                return;
            }

            var root = new GameObject("GeneratedScene");
            Undo.RegisterCreatedObjectUndo(root, "Import Generated Scene");

            // ---- terrain ---------------------------------------------
            if (!string.IsNullOrEmpty(manifest.terrain))
            {
                var terrainPath = assetsFolder + "/" + manifest.terrain;
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                    terrainPath);
                if (prefab != null)
                {
                    var terrain = (GameObject)PrefabUtility
                        .InstantiatePrefab(prefab, root.transform);
                    terrain.name = "Terrain";
                    if (_addColliders) AddColliders(terrain);
                    if (_staticBatching) MarkStatic(terrain);
                }
                else
                {
                    _status += "Terrain not found at " + terrainPath + "\n";
                }
            }

            // ---- load each distinct model once -------------------------
            var models = new Dictionary<string, GameObject>();
            foreach (var asset in manifest.assets)
            {
                var path = assetsFolder + "/" + asset.file;
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                if (prefab == null)
                {
                    _status += "Missing model: " + path + "\n";
                    continue;
                }
                models[asset.file] = prefab;
            }

            if (models.Count == 0)
            {
                _status += "\nNo models could be loaded. Is glTFast installed "
                           + "(com.unity.cloud.gltfast)? Without it Unity "
                           + "cannot read .glb files.";
                DestroyImmediate(root);
                return;
            }

            // Group by category so the hierarchy is navigable rather than a
            // flat list of two thousand objects.
            var groups = new Dictionary<string, Transform>();
            var offsets = new Dictionary<string, float>();
            foreach (var a in manifest.assets) offsets[a.file] = a.base_offset;

            int placed = 0, missing = 0;
            var mirror = _mirrorX ? -1f : 1f;

            try
            {
                for (int i = 0; i < manifest.instances.Length; i++)
                {
                    var inst = manifest.instances[i];

                    if (i % 64 == 0)
                    {
                        EditorUtility.DisplayProgressBar(
                            "Building scene",
                            $"{placed} / {manifest.instances.Length}",
                            (float)i / manifest.instances.Length);
                    }

                    if (!models.TryGetValue(inst.asset, out var prefab))
                    {
                        missing++;
                        continue;
                    }

                    if (!groups.TryGetValue(inst.category, out var parent))
                    {
                        var g = new GameObject(inst.category);
                        g.transform.SetParent(root.transform);
                        parent = g.transform;
                        groups[inst.category] = parent;
                    }

                    var go = (GameObject)PrefabUtility.InstantiatePrefab(
                        prefab, parent);
                    go.name = inst.name + "_" + placed;

                    var offset = offsets.TryGetValue(inst.asset, out var o)
                        ? o : 0f;

                    go.transform.localPosition = new Vector3(
                        inst.p[0] * mirror,
                        inst.p[1] - offset * inst.s,
                        inst.p[2]);
                    go.transform.localRotation = Quaternion.Euler(
                        0f, -inst.r * Mathf.Rad2Deg * mirror, 0f);
                    go.transform.localScale = Vector3.one * inst.s;

                    if (_addColliders && inst.category != "vegetation")
                        AddColliders(go);
                    if (_staticBatching) MarkStatic(go);

                    placed++;
                }
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }

            Selection.activeGameObject = root;
            SceneView.lastActiveSceneView?.FrameSelected();

            _status +=
                $"\nPlaced {placed} instances from {models.Count} models."
                + (missing > 0 ? $" {missing} could not be resolved." : "")
                + $"\nRendered triangles: {manifest.stats?.rendered_triangles:N0}";
        }

        // Colliders on every leaf of a big scene are expensive; vegetation is
        // skipped above because walking through undergrowth is expected.
        static void AddColliders(GameObject go)
        {
            foreach (var filter in go.GetComponentsInChildren<MeshFilter>())
            {
                if (filter.sharedMesh == null) continue;
                if (filter.GetComponent<Collider>() != null) continue;
                var collider = filter.gameObject.AddComponent<MeshCollider>();
                collider.sharedMesh = filter.sharedMesh;
            }
        }

        static void MarkStatic(GameObject go)
        {
            foreach (var t in go.GetComponentsInChildren<Transform>())
                GameObjectUtility.SetStaticEditorFlags(
                    t.gameObject, StaticEditorFlags.BatchingStatic
                                  | StaticEditorFlags.ContributeGI
                                  | StaticEditorFlags.OccluderStatic
                                  | StaticEditorFlags.OccludeeStatic);
        }

        // "C:/.../Assets/GeneratedScenes/x" -> "Assets/GeneratedScenes/x"
        static string ToProjectRelative(string absolute)
        {
            absolute = absolute.Replace('\\', '/');
            var root = Application.dataPath.Replace('\\', '/');
            if (!absolute.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                return null;
            return "Assets" + absolute.Substring(root.Length);
        }
    }
}
