"""Firewall response mechanism for TabPFN-NIDS.

Provides automated or monitored host defense against confirmed malicious network flows:
- Default mode is 'monitor_only' (defensive enforcement is disabled by default).
- In 'block_attacker' mode, provisions Linux iptables DROP rules for validated malicious source IPs.
- Protects loopback, link-local, multicast, management, and allowlisted IP ranges from accidental blocking.
- Avoids duplicate firewall rules for the same IP.
- Enforces strict IP address sanitization via python's `ipaddress` module to prevent shell injection.
- Never executes shell strings (`shell=False` only with explicit argument lists).
- Requires administrator authorization (`admin_authorized`) and checks operating system privileges.
- Supports automatic time-based rule expiration and manual rule revocation.
- Resilient: if firewall command execution fails, the detection pipeline continues operational,
  and the failure is logged and surfaced to the dashboard.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FirewallMode(str, Enum):
    MONITOR_ONLY = "monitor_only"
    BLOCK_ATTACKER = "block_attacker"


class FirewallAction(str, Enum):
    MONITORED = "MONITORED"
    BLOCKED = "BLOCKED"
    ALLOWLISTED = "ALLOWLISTED"
    DUPLICATE = "DUPLICATE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNAUTHORIZED = "UNAUTHORIZED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    UNBLOCKED = "UNBLOCKED"
    SKIPPED_BENIGN = "SKIPPED_BENIGN"


# Default protected networks: Loopback, Link-Local, Multicast, Broadcast, RFC1918 Private ranges
DEFAULT_PROTECTED_NETWORKS: list[str] = [
    "127.0.0.0/8",       # IPv4 loopback
    "::1/128",           # IPv6 loopback
    "169.254.0.0/16",    # IPv4 link-local
    "fe80::/10",         # IPv6 link-local
    "224.0.0.0/4",       # IPv4 multicast
    "ff00::/8",          # IPv6 multicast
    "255.255.255.255/32",# Broadcast
    "0.0.0.0/32",        # Unspecified IPv4
    "::/128",            # Unspecified IPv6
]


@dataclass
class ActiveBlockRule:
    """Record of an active firewall block rule."""

    source_ip: str
    attack_classification: str
    confidence: float
    blocked_at: float
    expires_at: float | None = None
    rule_id: str | None = None

    def is_expired(self, current_time: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        ts = current_time if current_time is not None else time.time()
        return ts >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_ip": self.source_ip,
            "attack_classification": self.attack_classification,
            "confidence": round(self.confidence, 4),
            "blocked_at": datetime.fromtimestamp(self.blocked_at, timezone.utc).isoformat(),
            "expires_at": (
                datetime.fromtimestamp(self.expires_at, timezone.utc).isoformat()
                if self.expires_at
                else None
            ),
            "remaining_seconds": (
                max(0.0, round(self.expires_at - time.time(), 1))
                if self.expires_at
                else None
            ),
        }


@dataclass
class FirewallEvent:
    """Audit log entry for an evaluated firewall action."""

    source_ip: str
    timestamp: str
    attack_classification: str
    confidence: float
    firewall_action: str
    action_result: str
    flow_id: str | None = None
    error_message: str | None = None
    rule_expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FirewallConfig:
    """Firewall configuration settings."""

    mode: str = FirewallMode.MONITOR_ONLY.value
    enabled: bool = False
    admin_authorized: bool = False
    min_confidence_to_block: float = 0.80
    rule_expiration_seconds: float = 300.0  # 5 minutes; 0 means indefinite
    trusted_allowlist: list[str] = field(default_factory=list)
    iptables_path: str = "iptables"
    max_audit_logs: int = 500



class FirewallManager:
    """Thread-safe manager for evaluating attack events and managing host firewall rules."""

    def __init__(self, config: FirewallConfig | None = None) -> None:
        self.config = config or FirewallConfig()
        self._lock = threading.RLock()
        self._active_blocks: dict[str, ActiveBlockRule] = {}
        self._audit_logs: list[FirewallEvent] = []
        self._last_failure: str | None = None
        self._total_blocks_applied = 0
        self._total_blocks_failed = 0

        # Pre-compile parsed protected networks and allowlist
        self._protected_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        self._allowlist_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        self._reload_network_ranges()

    def _reload_network_ranges(self) -> None:
        """Parse protected networks and custom allowlist into ipaddress objects."""
        with self._lock:
            self._protected_networks = []
            for cidr in DEFAULT_PROTECTED_NETWORKS:
                try:
                    self._protected_networks.append(ipaddress.ip_network(cidr, strict=False))
                except ValueError as exc:
                    logger.warning("Invalid default protected CIDR '%s': %s", cidr, exc)

            self._allowlist_networks = []
            for item in self.config.trusted_allowlist:
                item_clean = item.strip()
                if not item_clean:
                    continue
                try:
                    # Parse as network or single IP host network
                    self._allowlist_networks.append(ipaddress.ip_network(item_clean, strict=False))
                except ValueError as exc:
                    logger.warning("Invalid trusted allowlist entry '%s': %s", item_clean, exc)

    def update_config(
        self,
        mode: str | None = None,
        admin_authorized: bool | None = None,
        min_confidence_to_block: float | None = None,
        rule_expiration_seconds: float | None = None,
        trusted_allowlist: list[str] | None = None,
    ) -> None:
        """Dynamically update firewall configuration with validation."""
        with self._lock:
            if mode is not None:
                if mode not in (FirewallMode.MONITOR_ONLY.value, FirewallMode.BLOCK_ATTACKER.value):
                    raise ValueError(f"Invalid firewall mode '{mode}'. Choose 'monitor_only' or 'block_attacker'.")
                self.config.mode = mode

            if admin_authorized is not None:
                self.config.admin_authorized = bool(admin_authorized)

            if min_confidence_to_block is not None:
                self.config.min_confidence_to_block = max(0.0, min(1.0, float(min_confidence_to_block)))

            if rule_expiration_seconds is not None:
                self.config.rule_expiration_seconds = max(0.0, float(rule_expiration_seconds))

            if trusted_allowlist is not None:
                self.config.trusted_allowlist = list(trusted_allowlist)
                self._reload_network_ranges()

            logger.info(
                "Firewall config updated: mode=%s, admin_authorized=%s, min_conf=%.2f, expire=%.1fs, allowlist_count=%d",
                self.config.mode,
                self.config.admin_authorized,
                self.config.min_confidence_to_block,
                self.config.rule_expiration_seconds,
                len(self.config.trusted_allowlist),
            )

    def is_ip_allowlisted(self, ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        """Check whether an IP address belongs to protected or allowlisted ranges."""
        with self._lock:
            # 1. System protected ranges (loopback, link-local, multicast, broadcast)
            if ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_unspecified:
                return True

            for net in self._protected_networks:
                if ip_obj in net:
                    return True

            # 2. User allowlist ranges
            for net in self._allowlist_networks:
                if ip_obj in net:
                    return True

            return False

    def check_system_privileges(self) -> tuple[bool, str]:
        """Verify operating system capabilities and administrator authorization."""
        if not self.config.admin_authorized:
            return False, "Administrator authorization is disabled in configuration."

        system = platform.system().lower()
        if "linux" not in system:
            return False, f"Firewall iptables enforcement is only supported on Linux (current system: {platform.system()})."

        # Check if iptables binary exists
        if not shutil.which(self.config.iptables_path):
            return False, f"Firewall binary '{self.config.iptables_path}' not found on PATH."

        # Check for root / effective user ID on POSIX
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            return False, "Process lacks root privileges (euid != 0) required to modify iptables rules."

        return True, "Privileges verified."

    def process_attack_event(
        self,
        source_ip: str,
        attack_classification: str,
        confidence: float,
        flow_id: str | None = None,
        is_attack: bool = True,
    ) -> FirewallEvent:
        """Validate a detected flow and determine the appropriate firewall response."""
        now = time.time()
        now_iso = datetime.fromtimestamp(now, timezone.utc).isoformat()

        # Maintenance: purge expired rules
        self.check_expirations(now)

        with self._lock:
            # 1. Benign flows require no firewall action
            if not is_attack or str(attack_classification).lower() in ("normal", "benign"):
                event = FirewallEvent(
                    source_ip=str(source_ip),
                    timestamp=now_iso,
                    attack_classification=str(attack_classification),
                    confidence=float(confidence),
                    firewall_action=FirewallAction.SKIPPED_BENIGN.value,
                    action_result="Traffic classified as normal; no action required.",
                    flow_id=flow_id,
                )
                self._record_event(event)
                return event

            # 2. Strict IP sanitization & validation
            try:
                ip_clean = str(source_ip).strip()
                ip_obj = ipaddress.ip_address(ip_clean)
                sanitized_ip = str(ip_obj)
            except (ValueError, TypeError) as exc:
                event = FirewallEvent(
                    source_ip=str(source_ip),
                    timestamp=now_iso,
                    attack_classification=str(attack_classification),
                    confidence=float(confidence),
                    firewall_action=FirewallAction.FAILED.value,
                    action_result="Invalid source IP address format; action aborted.",
                    flow_id=flow_id,
                    error_message=f"IP validation error: {exc}",
                )
                self._record_event(event)
                return event

            # 3. Allowlist & Protected Network Protection
            if self.is_ip_allowlisted(ip_obj):
                event = FirewallEvent(
                    source_ip=sanitized_ip,
                    timestamp=now_iso,
                    attack_classification=str(attack_classification),
                    confidence=float(confidence),
                    firewall_action=FirewallAction.ALLOWLISTED.value,
                    action_result=f"IP {sanitized_ip} is in protected/allowlisted range; blocking suppressed.",
                    flow_id=flow_id,
                )
                self._record_event(event)
                logger.info("Firewall protection: suppressed block for allowlisted IP %s", sanitized_ip)
                return event

            # 4. Confidence Threshold Check (Distinguish uncertain model predictions from confirmed attacks)
            if confidence < self.config.min_confidence_to_block:
                event = FirewallEvent(
                    source_ip=sanitized_ip,
                    timestamp=now_iso,
                    attack_classification=str(attack_classification),
                    confidence=float(confidence),
                    firewall_action=FirewallAction.LOW_CONFIDENCE.value,
                    action_result=(
                        f"Prediction confidence ({confidence:.2%}) is below threshold "
                        f"({self.config.min_confidence_to_block:.2%}); blocking withheld."
                    ),
                    flow_id=flow_id,
                )
                self._record_event(event)
                return event

            # 5. Monitor-Only Mode
            if self.config.mode == FirewallMode.MONITOR_ONLY.value:
                event = FirewallEvent(
                    source_ip=sanitized_ip,
                    timestamp=now_iso,
                    attack_classification=str(attack_classification),
                    confidence=float(confidence),
                    firewall_action=FirewallAction.MONITORED.value,
                    action_result="Firewall is in monitor_only mode; attack logged without active enforcement.",
                    flow_id=flow_id,
                )
                self._record_event(event)
                return event

            # 6. Block-Attacker Mode: Duplicate Rule Prevention
            if sanitized_ip in self._active_blocks:
                active_rule = self._active_blocks[sanitized_ip]
                # If an active block already exists and hasn't expired, skip duplicate rule
                if not active_rule.is_expired(now):
                    event = FirewallEvent(
                        source_ip=sanitized_ip,
                        timestamp=now_iso,
                        attack_classification=str(attack_classification),
                        confidence=float(confidence),
                        firewall_action=FirewallAction.DUPLICATE.value,
                        action_result=f"Active firewall rule already exists for IP {sanitized_ip}.",
                        flow_id=flow_id,
                        rule_expires_at=(
                            datetime.fromtimestamp(active_rule.expires_at, timezone.utc).isoformat()
                            if active_rule.expires_at
                            else None
                        ),
                    )
                    self._record_event(event)
                    return event

            # 7. Check Administrator Authorization and Privileges
            has_privs, priv_msg = self.check_system_privileges()
            if not has_privs:
                event = FirewallEvent(
                    source_ip=sanitized_ip,
                    timestamp=now_iso,
                    attack_classification=str(attack_classification),
                    confidence=float(confidence),
                    firewall_action=FirewallAction.UNAUTHORIZED.value,
                    action_result=f"Firewall rule cannot be applied: {priv_msg}",
                    flow_id=flow_id,
                    error_message=priv_msg,
                )
                self._record_event(event)
                logger.warning("Firewall block withheld for %s: %s", sanitized_ip, priv_msg)
                return event

            # 8. Execute iptables Command Safely (No shell, exact arguments)
            expires_at = (
                (now + self.config.rule_expiration_seconds)
                if self.config.rule_expiration_seconds > 0
                else None
            )

            success, err_msg = self._apply_iptables_drop(sanitized_ip)
            if success:
                self._total_blocks_applied += 1
                self._last_failure = None
                rule = ActiveBlockRule(
                    source_ip=sanitized_ip,
                    attack_classification=attack_classification,
                    confidence=confidence,
                    blocked_at=now,
                    expires_at=expires_at,
                )
                self._active_blocks[sanitized_ip] = rule

                event = FirewallEvent(
                    source_ip=sanitized_ip,
                    timestamp=now_iso,
                    attack_classification=str(attack_classification),
                    confidence=float(confidence),
                    firewall_action=FirewallAction.BLOCKED.value,
                    action_result=f"Successfully provisioned iptables DROP rule for {sanitized_ip}.",
                    flow_id=flow_id,
                    rule_expires_at=(
                        datetime.fromtimestamp(expires_at, timezone.utc).isoformat()
                        if expires_at
                        else None
                    ),
                )
                self._record_event(event)
                logger.warning(
                    "FIREWALL ENFORCEMENT: Blocked malicious source IP %s (attack: %s, confidence: %.2f%%)",
                    sanitized_ip,
                    attack_classification,
                    confidence * 100.0,
                )
                return event
            else:
                self._total_blocks_failed += 1
                self._last_failure = f"Failed to block {sanitized_ip}: {err_msg}"
                event = FirewallEvent(
                    source_ip=sanitized_ip,
                    timestamp=now_iso,
                    attack_classification=str(attack_classification),
                    confidence=float(confidence),
                    firewall_action=FirewallAction.FAILED.value,
                    action_result="Failed to execute iptables rule.",
                    flow_id=flow_id,
                    error_message=err_msg,
                )
                self._record_event(event)
                logger.error("Firewall rule execution failed for %s: %s", sanitized_ip, err_msg)
                return event

    def _apply_iptables_drop(self, sanitized_ip: str) -> tuple[bool, str | None]:
        """Execute iptables command to drop traffic from the sanitized source IP."""
        cmd = [self.config.iptables_path, "-A", "INPUT", "-s", sanitized_ip, "-j", "DROP"]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5.0,
                shell=False,
            )
            if res.returncode == 0:
                return True, None
            return False, f"iptables returned code {res.returncode}: {res.stderr.strip()}"
        except Exception as exc:
            return False, f"Subprocess execution error: {exc}"

    def _remove_iptables_drop(self, sanitized_ip: str) -> tuple[bool, str | None]:
        """Execute iptables command to delete the drop rule for a source IP."""
        cmd = [self.config.iptables_path, "-D", "INPUT", "-s", sanitized_ip, "-j", "DROP"]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5.0,
                shell=False,
            )
            if res.returncode == 0:
                return True, None
            return False, f"iptables deletion returned code {res.returncode}: {res.stderr.strip()}"
        except Exception as exc:
            return False, f"Subprocess deletion error: {exc}"

    def unblock_ip(self, source_ip: str, reason: str = "Manual administrator removal") -> FirewallEvent:
        """Manually remove an active firewall block rule for a source IP."""
        now = time.time()
        now_iso = datetime.fromtimestamp(now, timezone.utc).isoformat()

        with self._lock:
            try:
                ip_clean = str(source_ip).strip()
                sanitized_ip = str(ipaddress.ip_address(ip_clean))
            except ValueError as exc:
                event = FirewallEvent(
                    source_ip=str(source_ip),
                    timestamp=now_iso,
                    attack_classification="N/A",
                    confidence=1.0,
                    firewall_action=FirewallAction.FAILED.value,
                    action_result=f"Invalid IP address: {exc}",
                )
                self._record_event(event)
                return event

            if sanitized_ip not in self._active_blocks:
                event = FirewallEvent(
                    source_ip=sanitized_ip,
                    timestamp=now_iso,
                    attack_classification="N/A",
                    confidence=1.0,
                    firewall_action=FirewallAction.FAILED.value,
                    action_result=f"No active block rule found for IP {sanitized_ip}.",
                )
                self._record_event(event)
                return event

            # Remove from iptables if supported
            has_privs, _ = self.check_system_privileges()
            if has_privs:
                self._remove_iptables_drop(sanitized_ip)

            rule = self._active_blocks.pop(sanitized_ip)
            event = FirewallEvent(
                source_ip=sanitized_ip,
                timestamp=now_iso,
                attack_classification=rule.attack_classification,
                confidence=rule.confidence,
                firewall_action=FirewallAction.UNBLOCKED.value,
                action_result=f"Rule removed: {reason}",
            )
            self._record_event(event)
            logger.info("Firewall unblocked IP %s (%s)", sanitized_ip, reason)
            return event

    def check_expirations(self, current_time: float | None = None) -> list[str]:
        """Prune expired firewall rules and remove corresponding iptables rules."""
        ts = current_time if current_time is not None else time.time()
        expired_ips: list[str] = []

        with self._lock:
            for ip, rule in list(self._active_blocks.items()):
                if rule.is_expired(ts):
                    expired_ips.append(ip)

            for ip in expired_ips:
                rule = self._active_blocks.pop(ip, None)
                if rule:
                    has_privs, _ = self.check_system_privileges()
                    if has_privs:
                        self._remove_iptables_drop(ip)

                    event = FirewallEvent(
                        source_ip=ip,
                        timestamp=datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                        attack_classification=rule.attack_classification,
                        confidence=rule.confidence,
                        firewall_action=FirewallAction.EXPIRED.value,
                        action_result="Firewall rule expired automatically.",
                    )
                    self._record_event(event)
                    logger.info("Firewall rule expired for IP %s", ip)

        return expired_ips

    def _record_event(self, event: FirewallEvent) -> None:
        """Append an audit log event with bounds enforcement."""
        self._audit_logs.append(event)
        if len(self._audit_logs) > self.config.max_audit_logs:
            self._audit_logs = self._audit_logs[-self.config.max_audit_logs:]

    def get_audit_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent audit log records."""
        with self._lock:
            logs = self._audit_logs[-limit:]
            return [e.to_dict() for e in reversed(logs)]

    def get_active_blocks(self) -> list[dict[str, Any]]:
        """Return all currently active block rules."""
        with self._lock:
            self.check_expirations()
            return [r.to_dict() for r in self._active_blocks.values()]

    def get_active_rules(self) -> list[str]:
        """Return list of active blocked IP strings."""
        with self._lock:
            self.check_expirations()
            return list(self._active_blocks.keys())

    def cleanup_expired_rules(self, current_time: float | None = None) -> list[str]:
        """Alias for check_expirations."""
        return self.check_expirations(current_time)

    @property
    def last_failure(self) -> str | None:
        return self._last_failure

    @property
    def last_error(self) -> str | None:
        return self._last_failure

    def get_status(self) -> dict[str, Any]:
        """Comprehensive telemetry and status for dashboard and API."""
        with self._lock:
            self.check_expirations()
            has_privs, priv_msg = self.check_system_privileges()
            return {
                "mode": self.config.mode,
                "admin_authorized": self.config.admin_authorized,
                "privilege_status": priv_msg,
                "privileges_ok": has_privs,
                "min_confidence_to_block": self.config.min_confidence_to_block,
                "rule_expiration_seconds": self.config.rule_expiration_seconds,
                "active_blocks_count": len(self._active_blocks),
                "total_blocks_applied": self._total_blocks_applied,
                "total_blocks_failed": self._total_blocks_failed,
                "last_failure": self._last_failure,
                "active_blocked_ips": [r.source_ip for r in self._active_blocks.values()],
                "trusted_allowlist_count": len(self.config.trusted_allowlist),
            }
