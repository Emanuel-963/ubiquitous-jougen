"""Tests for the multi-tier LicenseManager (LIC-01 / LIC-02 / LIC-03)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.license_manager import (
    FREE_FILE_LIMIT,
    LAB_SEAT_LIMIT,
    LicenseLimitError,
    LicenseManager,
    generate_key,
    key_tier,
    validate_key,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the LicenseManager singleton and prevent disk key loading."""
    LicenseManager.reset()
    # Patch persistence so tests never pick up a real key from ~/.ionflow/
    with patch.object(
        LicenseManager, "_load_persisted_key", return_value=None
    ), patch.object(LicenseManager, "_persist_key", return_value=None), patch.object(
        LicenseManager, "_delete_persisted_key", return_value=None
    ):
        LicenseManager.reset()  # reset again so new instance uses patched method
        yield
    LicenseManager.reset()


# ---------------------------------------------------------------------------
# generate_key
# ---------------------------------------------------------------------------


class TestGenerateKey:
    def test_pro_key_format(self):
        key = generate_key("CUST0001", "PRO")
        assert key.startswith("IONFLOW-PRO-CUST0001-")
        # Key: IONFLOW-PRO-CUST0001-MAC10 → 4 dash-separated parts
        parts = key.split("-")
        assert len(parts) == 4  # IONFLOW, PRO, CUST0001, MAC10

    def test_lab_key_format(self):
        key = generate_key("CUST0001", "LAB")
        assert key.startswith("IONFLOW-LAB-CUST0001-")

    def test_oem_key_format(self):
        key = generate_key("CUST0001", "OEM")
        assert key.startswith("IONFLOW-OEM-CUST0001-")

    def test_default_tier_is_pro(self):
        key = generate_key("ABCD1234")
        assert key.startswith("IONFLOW-PRO-")

    def test_lowercase_serial_is_folded(self):
        key = generate_key("cust0001", "PRO")
        assert "CUST0001" in key

    def test_invalid_serial_length(self):
        with pytest.raises(ValueError, match="exactly"):
            generate_key("SHORT")

    def test_invalid_serial_chars(self):
        with pytest.raises(ValueError, match="A-Z and 0-9"):
            generate_key("CUST-001")

    def test_unknown_tier(self):
        with pytest.raises(ValueError, match="Unknown tier"):
            generate_key("CUST0001", "GOLD")


# ---------------------------------------------------------------------------
# validate_key / key_tier
# ---------------------------------------------------------------------------


class TestValidateKey:
    def test_valid_pro_key(self):
        key = generate_key("ABCD1234", "PRO")
        assert validate_key(key) is True

    def test_valid_lab_key(self):
        key = generate_key("ABCD1234", "LAB")
        assert validate_key(key) is True

    def test_valid_oem_key(self):
        key = generate_key("ABCD1234", "OEM")
        assert validate_key(key) is True

    def test_invalid_key_returns_false(self):
        assert validate_key("IONFLOW-PRO-ABCD1234-0000000000") is False

    def test_empty_string(self):
        assert validate_key("") is False

    def test_garbage_string(self):
        assert validate_key("not-a-key") is False

    def test_key_tier_pro(self):
        key = generate_key("ABCD1234", "PRO")
        assert key_tier(key) == "PRO"

    def test_key_tier_lab(self):
        key = generate_key("ABCD1234", "LAB")
        assert key_tier(key) == "LAB"

    def test_key_tier_oem(self):
        key = generate_key("ABCD1234", "OEM")
        assert key_tier(key) == "OEM"

    def test_key_tier_invalid(self):
        assert key_tier("garbage") is None

    def test_pro_key_does_not_validate_as_lab(self):
        pro_key = generate_key("ABCD1234", "PRO")
        lab_key = generate_key("ABCD1234", "LAB")
        assert pro_key != lab_key
        # Swapping tier prefix breaks MAC
        tampered = pro_key.replace("IONFLOW-PRO-", "IONFLOW-LAB-")
        assert validate_key(tampered) is False


# ---------------------------------------------------------------------------
# LicenseManager — free tier
# ---------------------------------------------------------------------------


class TestFreeTier:
    def test_default_tier_is_free(self):
        mgr = LicenseManager.get()
        assert mgr.tier == "free"

    def test_is_pro_false_on_free(self):
        mgr = LicenseManager.get()
        assert mgr.is_pro is False

    def test_is_lab_false_on_free(self):
        mgr = LicenseManager.get()
        assert mgr.is_lab is False

    def test_is_oem_false_on_free(self):
        mgr = LicenseManager.get()
        assert mgr.is_oem is False

    def test_check_file_limit_within_free(self):
        mgr = LicenseManager.get()
        mgr.check_file_limit(FREE_FILE_LIMIT)  # should not raise

    def test_check_file_limit_exceeds_free(self):
        mgr = LicenseManager.get()
        with pytest.raises(LicenseLimitError):
            mgr.check_file_limit(FREE_FILE_LIMIT + 1)

    def test_status_label_free(self):
        mgr = LicenseManager.get()
        label = mgr.status_label()
        assert "gratuita" in label.lower() or "free" in label.lower()


# ---------------------------------------------------------------------------
# LicenseManager — Pro tier
# ---------------------------------------------------------------------------


class TestProTier:
    def test_activate_pro_key(self):
        mgr = LicenseManager.get()
        key = generate_key("ABCD1234", "PRO")
        assert mgr.activate(key) is True
        assert mgr.tier == "pro"

    def test_is_pro_after_activation(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("ABCD1234", "PRO"))
        assert mgr.is_pro is True

    def test_is_lab_false_for_pro(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("ABCD1234", "PRO"))
        assert mgr.is_lab is False

    def test_no_file_limit_on_pro(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("ABCD1234", "PRO"))
        mgr.check_file_limit(1000)  # should not raise

    def test_activate_invalid_key_returns_false(self):
        mgr = LicenseManager.get()
        assert mgr.activate("IONFLOW-PRO-XXXXXXXX-0000000000") is False
        assert mgr.tier == "free"

    def test_deactivate_reverts_to_free(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("ABCD1234", "PRO"))
        mgr.deactivate()
        assert mgr.tier == "free"

    def test_status_label_pro(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("ABCD1234", "PRO"))
        label = mgr.status_label()
        assert "pro" in label.lower()


# ---------------------------------------------------------------------------
# LicenseManager — Lab tier
# ---------------------------------------------------------------------------


class TestLabTier:
    def test_activate_lab_key(self):
        mgr = LicenseManager.get()
        key = generate_key("LAB00001", "LAB")
        assert mgr.activate(key) is True
        assert mgr.tier == "lab"

    def test_is_pro_true_for_lab(self):
        """Lab users get all Pro features (is_pro should return True)."""
        mgr = LicenseManager.get()
        mgr.activate(generate_key("LAB00001", "LAB"))
        assert mgr.is_pro is True

    def test_is_lab_true(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("LAB00001", "LAB"))
        assert mgr.is_lab is True

    def test_is_oem_false_for_lab(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("LAB00001", "LAB"))
        assert mgr.is_oem is False

    def test_lab_seat_limit_constant(self):
        assert LAB_SEAT_LIMIT == 5

    def test_status_label_lab(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("LAB00001", "LAB"))
        label = mgr.status_label()
        assert "lab" in label.lower()


# ---------------------------------------------------------------------------
# LicenseManager — OEM tier
# ---------------------------------------------------------------------------


class TestOEMTier:
    def test_activate_oem_key(self):
        mgr = LicenseManager.get()
        key = generate_key("OEM00001", "OEM")
        assert mgr.activate(key) is True
        assert mgr.tier == "oem"

    def test_is_pro_true_for_oem(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("OEM00001", "OEM"))
        assert mgr.is_pro is True

    def test_is_lab_true_for_oem(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("OEM00001", "OEM"))
        assert mgr.is_lab is True

    def test_is_oem_true(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("OEM00001", "OEM"))
        assert mgr.is_oem is True

    def test_status_label_oem(self):
        mgr = LicenseManager.get()
        mgr.activate(generate_key("OEM00001", "OEM"))
        label = mgr.status_label()
        assert "oem" in label.lower()
