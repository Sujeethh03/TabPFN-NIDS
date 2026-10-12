"""Network interface and TShark environment discovery on Windows."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Standard installation paths on Windows
COMMON_WINDOWS_TSHARK_PATHS = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Wireshark" / "tshark.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Wireshark" / "tshark.exe",
    Path(r"C:\Program Files\Wireshark\tshark.exe"),
    Path(r"C:\Program Files (x86)\Wireshark\tshark.exe"),
]


class TSharkNotFoundError(FileNotFoundError):
    """Raised when TShark binary is not found on the system."""

    def __init__(self, attempted_paths: list[str] | None = None) -> None:
        msg = (
            "TShark executable was not found. Please ensure Wireshark with TShark "
            "and Npcap is installed. On Windows, install from https://www.wireshark.org/ "
            "and ensure 'TShark' and 'Npcap' components are selected during installation. "
            "Alternatively, add Wireshark to your system PATH or set the TSHARK_PATH "
            "environment variable."
        )
        if attempted_paths:
            msg += f" (Attempted paths: {', '.join(attempted_paths)})"
        super().__init__(msg)


class NpcapNotFoundError(RuntimeError):
    """Raised when Npcap capture driver is not installed or available on Windows."""

    def __init__(self, detail: str = "") -> None:
        msg = (
            "Npcap packet capture driver is not installed or not running. "
            "Npcap is required for live network packet capture on Windows. "
            "Please download and install Npcap from https://npcap.com/#download "
            "with 'WinPcap API-compatible mode' checked, or re-run the Wireshark installer."
        )
        if detail:
            msg += f" Details: {detail}"
        super().__init__(msg)


class InterfaceNotFoundError(ValueError):
    """Raised when a requested network interface cannot be found."""

    def __init__(self, requested: str, available: list[str]) -> None:
        msg = (
            f"Interface '{requested}' was not found. "
            f"Available interfaces: {', '.join(available) if available else 'None detected'}. "
            "Run 'tshark -D' to list active network adapters."
        )
        super().__init__(msg)


@dataclass
class NetworkInterface:
    """Represents a discovered network interface."""

    index: int
    device: str
    name: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "device": self.device,
            "description": self.description,
        }


def find_tshark_binary(configured_path: str | Path | None = None) -> Path:
    """Locate the tshark executable on the system.

    Checks in order:
    1. Configured path (if provided).
    2. Environment variable ``TSHARK_PATH``.
    3. System PATH via ``shutil.which``.
    4. Common Windows installation directories.

    Returns:
        Absolute Path to the tshark executable.

    Raises:
        TSharkNotFoundError: If tshark cannot be located anywhere.
    """
    attempted: list[str] = []

    # 1. Configured path
    if configured_path:
        p = Path(configured_path).expanduser()
        attempted.append(str(p))
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
        # If it's just a bare command name (e.g. "tshark"), try which
        resolved = shutil.which(str(configured_path))
        if resolved:
            return Path(resolved).resolve()

    # 2. Environment variable
    env_path = os.environ.get("TSHARK_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        attempted.append(f"env TSHARK_PATH: {p}")
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()

    # 3. System PATH
    which_path = shutil.which("tshark")
    if which_path:
        return Path(which_path).resolve()
    attempted.append("System PATH (tshark)")

    # 4. Common Windows default locations
    if sys.platform == "win32":
        for default_path in COMMON_WINDOWS_TSHARK_PATHS:
            attempted.append(str(default_path))
            if default_path.is_file():
                return default_path.resolve()

    raise TSharkNotFoundError(attempted)


def check_npcap_installed() -> bool:
    """Check whether Npcap packet capture driver is installed on Windows.

    Returns:
        True if Npcap is detected or running on non-Windows; False if missing.
    """
    if sys.platform != "win32":
        return True

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    npcap_paths = [
        system_root / "System32" / "Npcap" / "wpcap.dll",
        system_root / "System32" / "wpcap.dll",
        system_root / "SysWOW64" / "Npcap" / "wpcap.dll",
        system_root / "SysWOW64" / "wpcap.dll",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Npcap",
    ]

    for p in npcap_paths:
        if p.exists():
            return True

    # Check via sc.exe query npcap service
    try:
        res = subprocess.run(
            ["sc", "query", "npcap"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if res.returncode == 0:
            return True
    except Exception:
        pass

    return False


def list_interfaces(tshark_path: str | Path | None = None) -> list[NetworkInterface]:
    """Discover network interfaces available to TShark.

    Executes ``tshark -D`` and parses each entry.
    Lines typically look like:
        1. \\Device\\NPF_{66E6F35E-...} (Wi-Fi)
        9. \\Device\\NPF_Loopback (Adapter for loopback traffic capture)

    Returns:
        List of NetworkInterface objects.
    """
    tshark_bin = find_tshark_binary(tshark_path)

    try:
        proc = subprocess.run(
            [str(tshark_bin), "-D"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("tshark -D timed out while discovering network interfaces.") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to execute tshark -D: {exc}") from exc

    if proc.returncode != 0:
        err = proc.stderr.strip()
        if "npcap" in err.lower() or "winpcap" in err.lower():
            raise NpcapNotFoundError(err)
        raise RuntimeError(f"tshark -D failed with exit code {proc.returncode}: {err}")

    interfaces: list[NetworkInterface] = []
    # Pattern: 1. \Device\NPF_{GUID} (Friendly Name) or 1. eth0
    line_pattern = re.compile(r"^\s*(\d+)\.\s+(\S+)(?:\s+\((.*)\))?\s*$")

    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = line_pattern.match(line)
        if match:
            idx = int(match.group(1))
            device = match.group(2)
            desc = (match.group(3) or "").strip()
            # If friendly description is present, prefer it for human name
            name = desc if desc else device
            interfaces.append(
                NetworkInterface(
                    index=idx,
                    device=device,
                    name=name,
                    description=desc or device,
                )
            )

    return interfaces


def resolve_interface(
    identifier: str | int | None,
    interfaces: list[NetworkInterface],
) -> NetworkInterface:
    """Resolve user-supplied interface identifier to a NetworkInterface.

    Can match:
    - Numeric index (e.g. 4 or "4")
    - Exact name (e.g. "Wi-Fi", case-insensitive)
    - Device string (e.g. "\\Device\\NPF_{...}")
    - Substring in name or description

    Returns:
        Matched NetworkInterface.

    Raises:
        InterfaceNotFoundError: If no matching interface is found.
    """
    if not interfaces:
        raise InterfaceNotFoundError(str(identifier), [])

    if identifier is None:
        raise InterfaceNotFoundError("None", [i.name for i in interfaces])

    ident_str = str(identifier).strip()

    # 1. Match by numeric index
    if ident_str.isdigit():
        idx = int(ident_str)
        for iface in interfaces:
            if iface.index == idx:
                return iface

    # 2. Match exact name (case-insensitive)
    for iface in interfaces:
        if iface.name.lower() == ident_str.lower():
            return iface

    # 3. Match exact device
    for iface in interfaces:
        if iface.device.lower() == ident_str.lower():
            return iface

    # 4. Match description (case-insensitive)
    for iface in interfaces:
        if iface.description.lower() == ident_str.lower():
            return iface

    # 5. Substring match
    for iface in interfaces:
        if ident_str.lower() in iface.name.lower() or ident_str.lower() in iface.description.lower():
            return iface

    available = [f"{i.index}: {i.name}" for i in interfaces]
    raise InterfaceNotFoundError(ident_str, available)
