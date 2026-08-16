# Setup Guide

Follow these in order. Everything here is free.

---

## 1. Install Python

You need **Python 3.10 or newer**. Check what you have:

```bash
python --version
```

If it's missing or older than 3.10, install from [python.org/downloads](https://www.python.org/downloads/).
On Windows, tick **"Add Python to PATH"** during installation.

---

## 2. Get the project running

```bash
cd text-to-3d-env-gen

# Create an isolated environment so this project's packages
# don't clash with anything else on your machine
python -m venv .venv

# Activate it
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

You'll need to run the activate command each time you open a new terminal.
You know it worked when your prompt shows `(.venv)`.

**Verify:**

```bash
python -m pytest tests/ -v
```

All tests should pass. If they do, the core pipeline is working.

---

## 3. Install Ollama (the local LLM)

This is what replaces a paid API. It runs the language model on your own GPU.

1. Download from **[ollama.com/download](https://ollama.com/download)** and install.
2. Ollama usually starts automatically. If not, run `ollama serve` in a terminal and leave it open.
3. Pull a model — **which one depends on your GPU's VRAM:**

```bash
# Check your GPU first
nvidia-smi
```

Look at the total memory figure, then pick:

| Your VRAM | Pull this | Command |
|---|---|---|
| 8 GB or more | Qwen2.5 7B | `ollama pull qwen2.5:7b-instruct` |
| 6 GB | Llama 3.2 3B | `ollama pull llama3.2:3b-instruct-q4_K_M` |
| 4 GB | Qwen2.5 3B | `ollama pull qwen2.5:3b-instruct-q4_K_M` |
| No GPU / unsure | Qwen2.5 3B | `ollama pull qwen2.5:3b-instruct-q4_K_M` |

The download is a few GB and takes a while. It's a one-time thing.

**Verify:**

```bash
python -m src.cli --check
```

You want to see `Status: READY`.

If you pulled a model other than the 7B default, tell the pipeline about it:

```bash
python -m src.cli "a medieval village" --model llama3.2:3b-instruct-q4_K_M
```

---

## 4. Try it

```bash
# With the LLM
python -m src.cli "a foggy medieval village at dusk with a blacksmith forge"

# Without the LLM (keyword fallback — works anywhere)
python -m src.cli "a dense dark forest camp at night" --fallback
```

Compare the two outputs. The LLM version should infer objects that aren't
literally named in the prompt — that difference is worth a screenshot for
your Tuesday progress report.

---

## Troubleshooting

**`python: command not found`**
Try `python3` instead of `python`, and `pip3` instead of `pip`.

**`ModuleNotFoundError: No module named 'src'`**
You're in the wrong folder. `cd` into `text-to-3d-env-gen` (the folder containing `src/`) and run commands from there.

**`--check` says NOT REACHABLE**
Ollama isn't running. Open a terminal and run `ollama serve`, leave it open, try again.

**Model responses are very slow**
You're likely running on CPU rather than GPU. Confirm with `nvidia-smi` while a prompt is running — if GPU usage stays at 0%, Ollama isn't using the GPU. A smaller model from the table above will help either way.

**Out of memory errors**
Use a smaller model. Drop one row down in the VRAM table.

---

## What's committed vs. what isn't

Large files are deliberately excluded via `.gitignore` — don't force-add them:

- 3D assets → downloaded via script (week 2)
- Generated scenes → regenerated from prompts
- Unity `Library/` folder → rebuilt by Unity automatically

If a teammate clones the repo, they run this setup guide and the download
script, and end up with an identical working copy.
