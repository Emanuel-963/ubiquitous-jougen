"""Tests for src.metadata — filename-based metadata extraction."""

from src.metadata import extract_metadata


class TestExtractMetadata:
    def test_electrolyte_h2so4(self):
        elec, _, _ = extract_metadata("1 Nb2 GCT H2SO4.txt")
        assert elec == "H2SO4"

    def test_electrolyte_naoh(self):
        elec, _, _ = extract_metadata("1 Nb2 S316 NaOH Am1 Pani.txt")
        assert elec == "NaOH"

    def test_electrolyte_na2so4(self):
        elec, _, _ = extract_metadata("1 Nb2 S316 Na2SO4 Am2.txt")
        assert elec == "Na2SO4"

    def test_electrolyte_li2so4(self):
        elec, _, _ = extract_metadata("1 Nb4 1A Li2SO4 Saturado.txt")
        assert elec == "Li2SO4"

    def test_electrolyte_licl(self):
        elec, _, _ = extract_metadata("1 Nb4 1A LiCl saturado.txt")
        assert elec == "LiCl"

    def test_electrolyte_unknown(self):
        elec, _, _ = extract_metadata("sample_no_info.txt")
        assert elec == "Unknown"

    def test_current_01a(self):
        _, cur, _ = extract_metadata("1 GCD 0.1A 2000C Am1.txt")
        assert cur == "0.1A"

    def test_current_1a(self):
        _, cur, _ = extract_metadata("1 Nb4 1A Li2SO4 Saturado.txt")
        assert cur == "1A"

    def test_current_unknown(self):
        _, cur, _ = extract_metadata("sample_no_current.txt")
        assert cur == "Unknown"

    def test_treatment_gct(self):
        _, _, treat = extract_metadata("1 Nb2 GCT H2SO4.txt")
        assert treat == "GCT"

    def test_treatment_gc(self):
        _, _, treat = extract_metadata("1 Nb4 GC 0.1A H2SO4.txt")
        assert treat == "GC"

    def test_treatment_steel316(self):
        _, _, treat = extract_metadata("1 Nb2 S316 Na2SO4 Am4.txt")
        assert treat == "Steel316"

    def test_treatment_none(self):
        _, _, treat = extract_metadata("sample_no_info.txt")
        assert treat == "None"
