"""
CLI entry point for ``chest-xray-diagnosis``.

Commands::

    chest-xray-diagnosis serve           # Start the API server
    chest-xray-diagnosis analyze <path>  # Analyze a single image
    chest-xray-diagnosis setup           # Configure API keys interactively
    chest-xray-diagnosis download-models # Pre-download model weights
    chest-xray-diagnosis --version       # Show version
"""

import argparse
import json
import sys

import chexray_diagnosis


def cmd_serve(args):
    """Start the FastAPI server."""
    from chexray_diagnosis.server import create_app
    import uvicorn

    app = create_app()
    print(f"\n🚀 Starting Chest X-Ray Diagnosis API on port {args.port}...")
    print(f"   Docs: http://localhost:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_analyze(args):
    """Analyze a single chest X-ray image."""
    from chexray_diagnosis import ChestXRayAnalyzer

    analyzer = ChestXRayAnalyzer()
    result = analyzer.analyze(args.image)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"  Chest X-Ray Analysis Result")
        print(f"{'='*60}")
        print(f"  Status    : {result['clinical_summary']}")
        print(f"  Level     : {result['assessment_level']}")
        print(f"  Normal    : {result['is_normal']}")
        print(f"  Time      : {result['processing_time_ms']:.0f} ms")

        if result.get("pathologies"):
            print(f"\n  {'Pathology':<22} {'Prob':>6}  {'Confidence':<10} {'Decision'}")
            print(f"  {'-'*22} {'-'*6}  {'-'*10} {'-'*10}")
            for p in result["pathologies"]:
                print(f"  {p['pathology']:<22} {p['probability']:>5.1%}  {p['confidence_level']:<10} {p['clinical_decision']}")

        if result.get("ai_explanation"):
            print(f"\n  AI Explanation ({result.get('api_used', 'N/A')}):")
            print(f"  {'-'*50}")
            for line in result["ai_explanation"].split("\n")[:10]:
                print(f"  {line}")
            if len(result["ai_explanation"].split("\n")) > 10:
                print(f"  ... (truncated, use --json for full output)")

        print(f"\n{'='*60}\n")


def cmd_setup(_args):
    """Interactive API key setup."""
    from chexray_diagnosis.config import interactive_setup
    interactive_setup()


def cmd_download(_args):
    """Download model weights from HuggingFace Hub."""
    from chexray_diagnosis.models.download import download_models
    download_models(force=_args.force)


def main():
    parser = argparse.ArgumentParser(
        prog="chest-xray-diagnosis",
        description="AI-powered chest X-ray diagnostic assistant",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"chest-xray-diagnosis {chexray_diagnosis.__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a chest X-ray image")
    p_analyze.add_argument("image", help="Path to the X-ray image file")
    p_analyze.add_argument("--json", action="store_true", help="Output as JSON")

    # setup
    sub.add_parser("setup", help="Configure API keys interactively")

    # download-models
    p_dl = sub.add_parser("download-models", help="Pre-download model weights")
    p_dl.add_argument("--force", action="store_true", help="Re-download even if cached")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "setup":
        cmd_setup(args)
    elif args.command == "download-models":
        cmd_download(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
