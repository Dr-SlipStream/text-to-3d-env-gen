"""
The web app: type a place, get a playable, playtested 3D scene.

    python scripts/serve.py              then open http://localhost:8000
    python scripts/serve.py --offline    no language model (keyword parsing)
    python scripts/serve.py --port 8080

Standard library only -- no Flask, no build step. The page in web/app.html
talks to four endpoints:

    GET  /api/options?prompt=...    what the sentence sets (live, as you type)
    POST /api/generate              start a generation; returns a job id
    GET  /api/jobs/<id>             progress log, then the result
    GET  /api/scenes                every scene generated so far
    GET  /api/scenes/<slug>         one scene: summary + bot playtest

and generated scenes are served from /scenes/<slug>/, where each one's own
viewer (index.html) runs inside the app.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.pipeline import load_index, run                     # noqa: E402
from src.prompt_options import parse as parse_options        # noqa: E402

WEB = REPO / "web"

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("application/javascript", ".js")


class Jobs:
    """Generation jobs, run one at a time on a worker thread.

    One at a time because a generation already uses the GPU (embeddings,
    the language model) and the CPU (solver, RL bot) flat out; two at once
    would only make both slow.
    """

    def __init__(self, config):
        self.config = config
        self.jobs = {}
        self.lock = threading.Lock()
        self.gate = threading.Semaphore(1)

    def start(self, prompt: str) -> str:
        job_id = uuid.uuid4().hex[:10]
        job = {"id": job_id, "prompt": prompt, "status": "queued",
               "log": [], "result": None, "error": None,
               "started": time.time()}
        with self.lock:
            self.jobs[job_id] = job
        threading.Thread(target=self._work, args=(job,), daemon=True).start()
        return job_id

    def _work(self, job):
        with self.gate:
            job["status"] = "running"

            def log(message: str) -> None:
                job["log"].append(message)
                print("  " + message, flush=True)

            try:
                overrides = {"offline": True} if self.config.offline else {}
                result = run(job["prompt"], manifest=self.config.manifest,
                             out_root=self.config.out, overrides=overrides,
                             log=log)
                result.pop("export", None)
                job["result"] = result
                job["status"] = "done"
            except Exception as e:                       # report, don't die
                traceback.print_exc()
                job["error"] = f"{type(e).__name__}: {e}"
                job["status"] = "error"

    def get(self, job_id: str):
        with self.lock:
            return self.jobs.get(job_id)


def list_scenes(out_root: Path) -> list:
    scenes = []
    if not out_root.is_dir():
        return scenes
    for summary in out_root.glob("*/summary.json"):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data["modified"] = summary.stat().st_mtime
        data.pop("export", None)
        scenes.append(data)
    scenes.sort(key=lambda s: s["modified"], reverse=True)
    return scenes


def make_handler(config, jobs: Jobs):
    out_root = Path(config.out).resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "EnvGen/1.0"

        def log_message(self, fmt, *args):            # quiet access log
            pass

        # -- helpers -------------------------------------------------------
        def _json(self, data, status=200):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, root: Path):
            try:
                path = path.resolve()
                path.relative_to(root.resolve())     # no escaping the root
            except (ValueError, OSError):
                return self._json({"error": "not found"}, 404)
            if path.is_dir():
                path = path / "index.html"
            if not path.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(str(path))[0] or \
                "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        # -- routes ----------------------------------------------------------
        def do_GET(self):
            url = urlparse(self.path)
            path = unquote(url.path)

            if path in ("/", "/index.html"):
                return self._file(WEB / "app.html", WEB)
            if path.startswith("/web/"):
                return self._file(WEB / path[len("/web/"):], WEB)
            if path.startswith("/scenes/"):
                return self._file(out_root / path[len("/scenes/"):],
                                  out_root)

            if path == "/api/options":
                prompt = parse_qs(url.query).get("prompt", [""])[0]
                clean, opts = parse_options(prompt)
                if config.offline:
                    opts.offline = True
                return self._json({"clean_prompt": clean,
                                   "options": opts.to_dict()})
            if path == "/api/scenes":
                return self._json(list_scenes(out_root))
            if path.startswith("/api/scenes/"):
                slug = path[len("/api/scenes/"):].strip("/")
                folder = (out_root / slug).resolve()
                try:
                    folder.relative_to(out_root)
                except ValueError:
                    return self._json({"error": "not found"}, 404)
                summary = folder / "summary.json"
                if not summary.is_file():
                    return self._json({"error": "not found"}, 404)
                data = json.loads(summary.read_text(encoding="utf-8"))
                data.pop("export", None)
                play = folder / "playtest.json"
                data["playtest_detail"] = (
                    json.loads(play.read_text(encoding="utf-8"))
                    if play.is_file() else None)
                return self._json(data)
            if path.startswith("/api/jobs/"):
                job = jobs.get(path[len("/api/jobs/"):])
                if not job:
                    return self._json({"error": "no such job"}, 404)
                return self._json(job)
            if path == "/api/status":
                return self._json({"offline": config.offline,
                                   "library_ready": config.ready.is_set()})
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/generate":
                return self._json({"error": "not found"}, 404)
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "body must be JSON"}, 400)
            prompt = str(body.get("prompt", "")).strip()
            if not prompt:
                return self._json({"error": "Describe a place to generate."},
                                  400)
            if len(prompt) > 500:
                return self._json({"error": "Keep the description under "
                                            "500 characters."}, 400)
            return self._json({"job": jobs.start(prompt)}, 202)

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description="Run the generator web app")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--out", type=Path, default=REPO / "outputs")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--offline", action="store_true",
                   help="never call the language model")
    config = p.parse_args()
    config.out.mkdir(parents=True, exist_ok=True)
    config.ready = threading.Event()

    # Load the asset library (and its embedding model) up front, so the
    # first generation doesn't sit for ten seconds looking stuck.
    def warm():
        try:
            load_index(config.manifest)
        except Exception as e:
            print(f"  ! asset library not loaded: {e}")
        config.ready.set()
    threading.Thread(target=warm, daemon=True).start()

    jobs = Jobs(config)
    server = ThreadingHTTPServer((config.host, config.port),
                                 make_handler(config, jobs))
    print(f"\n  Environment generator running at "
          f"http://{config.host}:{config.port}\n"
          "  Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
