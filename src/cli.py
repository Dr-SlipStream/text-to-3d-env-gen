"""Command-line entry point for testing the pipeline stage by stage.

Usage:
    python -m src.cli "a foggy medieval village at dusk"
    python -m src.cli "a desert outpost" --fallback     # skip the LLM
    python -m src.cli --check                            # environment check
"""

from __future__ import annotations

import argparse
import sys

from .llm_client import LLMClient
from .prompt_decomposition import decompose


def cmd_check() -> int:
    """Report whether the local LLM is reachable. Run this first after setup."""
    client = LLMClient()
    print(f"Ollama host : {client.host}")
    print(f"Wanted model: {client.model}")

    models = client.available_models()
    if not models:
        print("\nStatus      : NOT REACHABLE")
        print("\nOllama isn't responding. Fix with:")
        print("  1. Install: https://ollama.com/download")
        print("  2. Start  : ollama serve")
        print(f"  3. Pull   : ollama pull {client.model}")
        print("\nThe pipeline still works without it (keyword fallback),")
        print("but prompt understanding will be much weaker.")
        return 1

    print(f"Installed   : {', '.join(models)}")
    if client.is_available():
        print("\nStatus      : READY")
        return 0

    print("\nStatus      : SERVER UP, MODEL MISSING")
    print(f"Pull it with: ollama pull {client.model}")
    return 1


def cmd_parse(prompt: str, fallback: bool, model: str | None) -> int:
    client = LLMClient(model=model) if model else None
    spec = decompose(prompt, client=client, force_fallback=fallback)

    print(f'\nPrompt : "{prompt}"')
    print(f"Parser : {spec.parser}")
    print(f"Summary: {spec.summary()}\n")

    if spec.warnings:
        for w in spec.warnings:
            print(f"  ! {w}")
        print()

    print(spec.to_json())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Text-to-3D environment pipeline")
    p.add_argument("prompt", nargs="?", help="environment description")
    p.add_argument("--fallback", action="store_true",
                   help="skip the LLM and use keyword parsing only")
    p.add_argument("--model", help="override the Ollama model name")
    p.add_argument("--check", action="store_true",
                   help="check whether the local LLM is set up correctly")
    args = p.parse_args(argv)

    if args.check:
        return cmd_check()
    if not args.prompt:
        p.print_help()
        return 1
    return cmd_parse(args.prompt, args.fallback, args.model)


if __name__ == "__main__":
    sys.exit(main())
