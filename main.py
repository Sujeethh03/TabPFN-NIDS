"""Command-line entry point for TabPFN-NIDS.

Primary mode  : Web server with PCAP file upload and analysis dashboard (default).
Secondary mode: Offline CLI PCAP analysis (--pcap flag, no web server).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TabPFN-NIDS Network Intrusion Detection System\n\n"
            "Default: starts the PCAP analysis web server at http://127.0.0.1:8000/\n"
            "Use --pcap <file> for headless CLI analysis without the web server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pcap",
        type=Path,
        default=None,
        help="Path to a PCAP file for offline CLI analysis (no web server).",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the web server (default: 127.0.0.1).",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web server (default: 8000).",
    )

    # Kept for backward compatibility with live-capture usage.
    parser.add_argument(
        "--interface",
        type=str,
        default=None,
        help="Network interface for live capture (legacy; exposed via the API).",
    )

    return parser.parse_args()


def run_offline_pcap(pcap_path: Path) -> int:
    """Run offline PCAP analysis from the CLI (no web server)."""
    pcap_path = pcap_path.resolve()

    if not pcap_path.exists():
        print(f"ERROR: PCAP file not found: {pcap_path}")
        return 1

    if pcap_path.suffix.lower() not in (".pcap", ".pcapng"):
        print(f"WARNING: File extension is not .pcap or .pcapng: {pcap_path}")

    analyze_script = PROJECT_ROOT / "scripts" / "analyze_pcap.py"

    if not analyze_script.exists():
        print(f"ERROR: Analysis script not found: {analyze_script}")
        return 1

    command = [
        sys.executable,
        str(analyze_script),
        "--pcap",
        str(pcap_path),
    ]

    print("=" * 70)
    print("TabPFN-NIDS — Offline PCAP Analysis (CLI)")
    print("=" * 70)
    print(f"PCAP   : {pcap_path}")
    print("Tip    : Run without --pcap to use the web dashboard instead.")
    print()

    result = subprocess.run(command, cwd=PROJECT_ROOT)
    return result.returncode


def run_server(host: str = "127.0.0.1", port: int = 8000, interface: str | None = None) -> int:
    """Start the FastAPI/Uvicorn server (PCAP analysis dashboard + live capture API)."""
    import uvicorn
    import os

    if interface:
        os.environ["CAPTURE_INTERFACE"] = interface

    print("=" * 70)
    print("TabPFN-NIDS — Web Server Starting")
    print("=" * 70)
    print(f"  Dashboard  : http://{host}:{port}/")
    print(f"  API Docs   : http://{host}:{port}/docs")
    print()
    print("  PCAP Analysis Endpoints:")
    print(f"    Upload   → POST http://{host}:{port}/api/pcap/upload")
    print(f"    Analyze  → POST http://{host}:{port}/api/pcap/analyze/{{job_id}}")
    print(f"    Status   → GET  http://{host}:{port}/api/pcap/status/{{job_id}}")
    print(f"    Report   → GET  http://{host}:{port}/api/pcap/report/{{job_id}}")
    print(f"    Download → GET  http://{host}:{port}/api/pcap/download/{{job_id}}")
    if interface:
        print(f"\n  Live Capture Interface: {interface}")
    print("=" * 70)
    print()

    uvicorn.run("tabpfn_nids.api.main:app", host=host, port=port, log_level="info")
    return 0


def main() -> int:
    args = _parse_args()

    if args.pcap:
        return run_offline_pcap(args.pcap)

    # Default: start the PCAP analysis web server
    return run_server(host=args.host, port=args.port, interface=args.interface)


if __name__ == "__main__":
    raise SystemExit(main())