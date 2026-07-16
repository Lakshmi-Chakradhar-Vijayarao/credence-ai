"""
credence_runtime/__main__.py — Credence CLI

Usage:
    credence calibrate --model Qwen/Qwen2.5-3B-Instruct --output cal.json
    credence tag       --model Qwen/Qwen2.5-3B-Instruct --calibration cal.json --prompt "Who wrote Hamlet?"
    credence serve     --model Qwen/Qwen2.5-3B-Instruct --calibration cal.json
    credence version

Or via python -m:
    python -m credence_runtime calibrate ...
"""

from __future__ import annotations

import sys

from esm.__main__ import build_parser, cmd_calibrate, cmd_tag


def cmd_serve(args):
    """Start the Credence HTTP server (FastAPI)."""
    try:
        from epistemic_runtime.serve import run_server
        run_server(
            model_id=args.model,
            calibration_path=args.calibration,
            host=args.host,
            port=args.port,
            device=args.device,
            dtype=args.dtype,
        )
    except ImportError:
        print(
            "ERROR: serve dependencies not installed.\n"
            "Run: pip install 'credence-runtime[serve]'",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_version(_args):
    from credence_runtime import __version__
    print(f"credence-runtime {__version__}")
    print("Credence AI — Cognitive Infrastructure for AI inference")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="credence",
        description="Credence — Cognitive Infrastructure for AI inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-time calibration (creates the Fisher probe)
  credence calibrate --model Qwen/Qwen2.5-3B-Instruct \\
                     --dataset trivia_qa --n 200 \\
                     --output cal.json

  # Tag a single prompt
  credence tag --model Qwen/Qwen2.5-3B-Instruct \\
               --calibration cal.json \\
               --prompt "Who wrote Hamlet?"

  # Start HTTP server
  credence serve --model Qwen/Qwen2.5-3B-Instruct --calibration cal.json

  # Check version
  credence version
""",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── calibrate ──────────────────────────────────────────────────────────────
    # Reuse the full argument spec from esm.__main__
    _esm_parser = build_parser()
    _esm_sub    = {a.prog.split()[-1]: a for a in _esm_parser._subparsers._actions[-1]._name_parser_map.values()}

    p_cal = sub.add_parser("calibrate", help="Calibrate Fisher probe via bilateral oracle",
                           parents=[_esm_sub.get("calibrate", argparse.ArgumentParser())],
                           add_help=False)
    p_cal.set_defaults(func=cmd_calibrate)

    p_tag = sub.add_parser("tag", help="Tag a single prompt and print EpistemicTag as JSON",
                           parents=[_esm_sub.get("tag", argparse.ArgumentParser())],
                           add_help=False)
    p_tag.set_defaults(func=cmd_tag)

    # ── serve ──────────────────────────────────────────────────────────────────
    p_srv = sub.add_parser("serve", help="Start Credence HTTP server (FastAPI)")
    p_srv.add_argument("--model", "-m", required=True, help="HuggingFace model ID")
    p_srv.add_argument("--calibration", "-c", required=True, help="Calibration JSON path")
    p_srv.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_srv.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_srv.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p_srv.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p_srv.set_defaults(func=cmd_serve)

    # ── version ────────────────────────────────────────────────────────────────
    p_ver = sub.add_parser("version", help="Print version and exit")
    p_ver.set_defaults(func=cmd_version)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
