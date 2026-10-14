"""Unit and integration tests for FirewallManager.

Tests:
1. monitor_only mode (no iptables execution)
2. block_attacker mode without admin authorization
3. Allowlist protection (loopback, private/management subnets)
4. Confidence threshold protection
5. Duplicate rule prevention
6. Safe execution without shell=True or string interpolation
7. Rule expiration and manual unblocking
8. Resilience when command execution fails (detection continues, failure logged)
"""

import time
from unittest.mock import MagicMock, patch
import pytest

from tabpfn_nids.firewall.firewall_manager import (
    FirewallManager,
    FirewallConfig,
    FirewallMode,
    FirewallAction,
)


@pytest.fixture
def firewall():
    config = FirewallConfig(
        mode=FirewallMode.MONITOR_ONLY.value,
        enabled=False,
        admin_authorized=False,
        rule_expiration_seconds=60,
        min_confidence_to_block=0.80,
    )
    return FirewallManager(config=config)


def test_monitor_only_mode(firewall):
    """In monitor_only mode, attacks are logged but iptables is never invoked."""
    event = firewall.process_attack_event(
        source_ip="198.51.100.22",
        attack_classification="DoS",
        confidence=0.95,
        flow_id="test_flow_1",
    )
    assert event.firewall_action == FirewallAction.MONITORED.value
    assert len(firewall.get_active_rules()) == 0
    assert len(firewall.get_audit_logs()) == 1


def test_block_attacker_unauthorized():
    """In block_attacker mode without explicit admin authorization, block is rejected."""
    config = FirewallConfig(
        mode=FirewallMode.BLOCK_ATTACKER.value,
        enabled=True,
        admin_authorized=False,
    )
    fw = FirewallManager(config=config)
    event = fw.process_attack_event(
        source_ip="198.51.100.22",
        attack_classification="Exploits",
        confidence=0.92,
    )
    assert event.firewall_action == FirewallAction.UNAUTHORIZED.value
    assert len(fw.get_active_rules()) == 0


def test_allowlist_protection_loopback():
    """Loopback addresses must never be blocked, regardless of attack classification."""
    config = FirewallConfig(
        mode=FirewallMode.BLOCK_ATTACKER.value,
        enabled=True,
        admin_authorized=True,
    )
    fw = FirewallManager(config=config)
    event = fw.process_attack_event(
        source_ip="127.0.0.1",
        attack_classification="DoS",
        confidence=0.99,
    )
    assert event.firewall_action == FirewallAction.ALLOWLISTED.value
    assert len(fw.get_active_rules()) == 0


def test_allowlist_custom_subnet():
    """Custom allowlisted management CIDR must be protected."""
    config = FirewallConfig(
        mode=FirewallMode.BLOCK_ATTACKER.value,
        enabled=True,
        admin_authorized=True,
        trusted_allowlist=["10.0.0.0/24", "192.168.1.50"],
    )
    fw = FirewallManager(config=config)

    # In allowlist CIDR
    event1 = fw.process_attack_event(
        source_ip="10.0.0.45",
        attack_classification="Reconnaissance",
        confidence=0.95,
    )
    assert event1.firewall_action == FirewallAction.ALLOWLISTED.value

    # Exact IP allowlisted
    event2 = fw.process_attack_event(
        source_ip="192.168.1.50",
        attack_classification="Reconnaissance",
        confidence=0.95,
    )
    assert event2.firewall_action == FirewallAction.ALLOWLISTED.value


def test_low_confidence_protection():
    """Low-confidence predictions below threshold must not trigger blocking."""
    config = FirewallConfig(
        mode=FirewallMode.BLOCK_ATTACKER.value,
        enabled=True,
        admin_authorized=True,
        min_confidence_to_block=0.85,
    )
    fw = FirewallManager(config=config)
    event = fw.process_attack_event(
        source_ip="203.0.113.15",
        attack_classification="Fuzzers",
        confidence=0.72,  # Below 0.85
    )
    assert event.firewall_action == FirewallAction.LOW_CONFIDENCE.value
    assert len(fw.get_active_rules()) == 0


def test_successful_block_and_duplicate_prevention():
    """Valid high-confidence attack triggers block; subsequent detections are marked DUPLICATE."""
    config = FirewallConfig(
        mode=FirewallMode.BLOCK_ATTACKER.value,
        enabled=True,
        admin_authorized=True,
        min_confidence_to_block=0.80,
    )
    fw = FirewallManager(config=config)

    with patch.object(fw, "check_system_privileges", return_value=(True, "Privileges verified")):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            # First detection -> BLOCKED
            event1 = fw.process_attack_event(
                source_ip="203.0.113.99",
                attack_classification="DoS",
                confidence=0.94,
            )
            assert event1.firewall_action == FirewallAction.BLOCKED.value
            assert "203.0.113.99" in fw.get_active_rules()

            # Verify iptables command used safe list args, no shell=True
            assert mock_run.call_count >= 1
            args, kwargs = mock_run.call_args
            assert args[0] == ["iptables", "-C", "INPUT", "-s", "203.0.113.99", "-j", "DROP"] or \
                   args[0] == ["iptables", "-A", "INPUT", "-s", "203.0.113.99", "-j", "DROP"]
            assert kwargs.get("shell") is not True

            # Second detection with same IP -> DUPLICATE
            event2 = fw.process_attack_event(
                source_ip="203.0.113.99",
                attack_classification="DoS",
                confidence=0.98,
            )
            assert event2.firewall_action == FirewallAction.DUPLICATE.value


def test_rule_expiration():
    """Expired rules are pruned and deleted from iptables."""
    config = FirewallConfig(
        mode=FirewallMode.BLOCK_ATTACKER.value,
        enabled=True,
        admin_authorized=True,
        rule_expiration_seconds=1.0,  # 1 second
    )
    fw = FirewallManager(config=config)

    with patch.object(fw, "check_system_privileges", return_value=(True, "Privileges verified")):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            fw.process_attack_event(
                source_ip="203.0.113.44",
                attack_classification="Exploits",
                confidence=0.95,
            )
            assert "203.0.113.44" in fw.get_active_rules()

            # Wait 1.1s for expiration
            time.sleep(1.1)

            expired = fw.cleanup_expired_rules()
            assert "203.0.113.44" in expired
            assert "203.0.113.44" not in fw.get_active_rules()


def test_firewall_failure_resilience():
    """If iptables execution fails with non-zero exit code or OS error, detection does not crash."""
    config = FirewallConfig(
        mode=FirewallMode.BLOCK_ATTACKER.value,
        enabled=True,
        admin_authorized=True,
    )
    fw = FirewallManager(config=config)

    with patch.object(fw, "check_system_privileges", return_value=(True, "Privileges verified")):
        with patch("subprocess.run", side_effect=OSError("iptables binary not found on Windows")):
            event = fw.process_attack_event(
                source_ip="198.51.100.77",
                attack_classification="Shellcode",
                confidence=0.96,
            )
            # Must return FAILED without re-raising exception
            assert event.firewall_action == FirewallAction.FAILED.value
            assert fw.last_error is not None
            assert "iptables binary not found" in fw.last_error
            assert "198.51.100.77" not in fw.get_active_rules()
