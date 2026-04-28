"""Tests for the potentiostat EIS file parsers (Phase 2).

Fixtures are embedded directly in the test module as string constants so
the tests have no external file dependencies and can run offline.

Each test follows the pattern:
    1. Write fixture bytes/text to a temporary file.
    2. Check ``can_parse()`` → True for the correct parser, False for others.
    3. Call ``parse()`` and verify the returned :class:`ParsedEIS`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.parsers import REGISTERED_PARSERS, detect_parser, parse_eis_file
from src.parsers.autolab import AutolabParser
from src.parsers.base import ParsedEIS, PotentiostatParser
from src.parsers.biologic import BioLogicParser
from src.parsers.gamry import GamryParser
from src.parsers.zahner import ZahnerParser

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _required_columns(df: pd.DataFrame) -> None:
    """Assert that df contains the three mandatory EIS columns."""
    for col in ("frequency", "zreal", "zimag"):
        assert col in df.columns, f"Missing column '{col}' in DataFrame"


def _valid_types(df: pd.DataFrame) -> None:
    """Assert that the three EIS columns are numeric."""
    for col in ("frequency", "zreal", "zimag"):
        assert pd.api.types.is_numeric_dtype(
            df[col]
        ), f"Column '{col}' is not numeric: {df[col].dtype}"


def _nonempty(df: pd.DataFrame) -> None:
    """Assert that the DataFrame has at least one row."""
    assert len(df) > 0, "DataFrame is empty"


# ---------------------------------------------------------------------------
# Gamry fixtures
# ---------------------------------------------------------------------------

GAMRY_DTA = textwrap.dedent(
    """\
    EXPLAIN
    TAG	GALEIS
    TITLE\tGalvanostatic EIS
    DATE\t01/01/2024
    TIME\t12:00:00
    PSTAT\tInterface 1010E

    ZCURVE\tTABLE
    Pt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\tIdc\tVdc\tIERange
    #\ts\tHz\tOhm\tOhm\tV\tOhm\tdegree\tA\tV\t#
    0\t0.0\t10000.0\t12.345\t-3.210\t0.001\t12.76\t-14.6\t0.0\t0.0\t0
    1\t0.1\t1000.0\t13.100\t-5.500\t0.001\t14.21\t-22.8\t0.0\t0.0\t0
    2\t0.2\t100.0\t15.200\t-8.900\t0.001\t17.66\t-30.3\t0.0\t0.0\t0
    """
)


@pytest.fixture()
def gamry_dta_file(tmp_path: Path) -> Path:
    f = tmp_path / "test_eis.dta"
    f.write_text(GAMRY_DTA, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# BioLogic fixtures
# ---------------------------------------------------------------------------

BIOLOGIC_MPT = textwrap.dedent(
    """\
    EC-Lab ASCII FILE
    Nb header lines : 13
    Device : SP-150
    Acquisition started on : 01/01/2024 12:00:00
    Saved on : 01/01/2024 12:05:00
    CE vs. WE compliance from -10 V to 10 V
    Channel : Analog 1
    Electrode material :
    Initial state :
    Electrolyte :
    Comments :
    Cycle Definition : Charge/Discharge alternance
    freq/Hz\tRe(Z)/Ohm\t-Im(Z)/Ohm\t|Z|/Ohm\tPhase(Z)/deg\ttime/s\t<Ewe>/V\t<I>/mA\tCs/µF\tCp/µF\tcycle number\tRe(Y)/S\tIm(Y)/S\t|Y|/S
    10000\t12.345\t3.210\t12.76\t-14.6\t0.0\t3.7\t0.1\t0\t0\t1\t0\t0\t0
    1000\t13.100\t5.500\t14.21\t-22.8\t0.1\t3.7\t0.1\t0\t0\t1\t0\t0\t0
    100\t15.200\t8.900\t17.66\t-30.3\t0.2\t3.7\t0.1\t0\t0\t1\t0\t0\t0
    """
)


@pytest.fixture()
def biologic_mpt_file(tmp_path: Path) -> Path:
    f = tmp_path / "test_eis.mpt"
    f.write_text(BIOLOGIC_MPT, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Autolab fixtures
# ---------------------------------------------------------------------------

AUTOLAB_CSV = textwrap.dedent(
    """\
    Autolab NOVA EIS Measurement
    Date: 01/01/2024
    Potentiostat: PGSTAT302N

    Frequency (Hz);Z' (Ohm);Z'' (Ohm);|Z| (Ohm);Phase (deg)
    10000;12.345;-3.210;12.76;-14.6
    1000;13.100;-5.500;14.21;-22.8
    100;15.200;-8.900;17.66;-30.3
    """
)


@pytest.fixture()
def autolab_csv_file(tmp_path: Path) -> Path:
    f = tmp_path / "test_eis.csv"
    f.write_text(AUTOLAB_CSV, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Zahner fixtures
# ---------------------------------------------------------------------------

ZAHNER_ISM = textwrap.dedent(
    """\
    [Impedance Spectrum]
    Instrument: Zennium E4
    Date: 01/01/2024
    Comment: Test spectrum

    Frequency[Hz]\tRe(Z)[Ohm]\tIm(Z)[Ohm]
    10000\t12.345\t-3.210
    1000\t13.100\t-5.500
    100\t15.200\t-8.900
    """
)


@pytest.fixture()
def zahner_ism_file(tmp_path: Path) -> Path:
    f = tmp_path / "test_eis.ism"
    f.write_text(ZAHNER_ISM, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Gamry tests
# ---------------------------------------------------------------------------


class TestGamryParser:
    def test_can_parse_correct_extension(self, gamry_dta_file: Path):
        assert GamryParser.can_parse(gamry_dta_file) is True

    def test_cannot_parse_csv(self, tmp_path: Path):
        f = tmp_path / "other.csv"
        f.write_text("freq,zreal,zimag\n1000,10,-5\n")
        assert GamryParser.can_parse(f) is False

    def test_parse_returns_parsed_eis(self, gamry_dta_file: Path):
        result = GamryParser().parse(gamry_dta_file)
        assert isinstance(result, ParsedEIS)
        assert "Gamry" in result.instrument

    def test_parse_dataframe_columns(self, gamry_dta_file: Path):
        df = GamryParser().parse(gamry_dta_file).data
        _required_columns(df)
        _valid_types(df)
        _nonempty(df)

    def test_parse_row_count(self, gamry_dta_file: Path):
        df = GamryParser().parse(gamry_dta_file).data
        assert len(df) == 3

    def test_parse_zimag_negative_convention(self, gamry_dta_file: Path):
        """zimag should be negative (capacitive arcs have -Im(Z) < 0)."""
        df = GamryParser().parse(gamry_dta_file).data
        # Gamry stores negative imaginary parts; fixture values are already negative
        assert (df["zimag"] <= 0).all()

    def test_validate_passes(self, gamry_dta_file: Path):
        """validate() must not raise for well-formed data."""
        result = GamryParser().parse(gamry_dta_file)
        result.validate()  # must not raise

    def test_source_file_recorded(self, gamry_dta_file: Path):
        result = GamryParser().parse(gamry_dta_file)
        assert result.source_file != ""


# ---------------------------------------------------------------------------
# BioLogic tests
# ---------------------------------------------------------------------------


class TestBioLogicParser:
    def test_can_parse_mpt(self, biologic_mpt_file: Path):
        assert BioLogicParser.can_parse(biologic_mpt_file) is True

    def test_cannot_parse_dta(self, gamry_dta_file: Path):
        assert BioLogicParser.can_parse(gamry_dta_file) is False

    def test_parse_returns_parsed_eis(self, biologic_mpt_file: Path):
        result = BioLogicParser().parse(biologic_mpt_file)
        assert isinstance(result, ParsedEIS)
        assert "BioLogic" in result.instrument

    def test_parse_dataframe_columns(self, biologic_mpt_file: Path):
        df = BioLogicParser().parse(biologic_mpt_file).data
        _required_columns(df)
        _valid_types(df)
        _nonempty(df)

    def test_parse_row_count(self, biologic_mpt_file: Path):
        df = BioLogicParser().parse(biologic_mpt_file).data
        assert len(df) == 3

    def test_zimag_negative_convention(self, biologic_mpt_file: Path):
        """BioLogic .mpt uses -Im(Z) column (already positive); parser should negate."""
        df = BioLogicParser().parse(biologic_mpt_file).data
        # After the sign-flip in _enforce_zimag_convention, values must be ≤ 0
        assert (df["zimag"] <= 0).all()

    def test_validate_passes(self, biologic_mpt_file: Path):
        result = BioLogicParser().parse(biologic_mpt_file)
        result.validate()


# ---------------------------------------------------------------------------
# Autolab tests
# ---------------------------------------------------------------------------


class TestAutolabParser:
    def test_can_parse_csv_with_autolab_header(self, autolab_csv_file: Path):
        assert AutolabParser.can_parse(autolab_csv_file) is True

    def test_cannot_parse_gamry_dta(self, gamry_dta_file: Path):
        assert AutolabParser.can_parse(gamry_dta_file) is False

    def test_parse_returns_parsed_eis(self, autolab_csv_file: Path):
        result = AutolabParser().parse(autolab_csv_file)
        assert isinstance(result, ParsedEIS)
        assert "Autolab" in result.instrument

    def test_parse_dataframe_columns(self, autolab_csv_file: Path):
        df = AutolabParser().parse(autolab_csv_file).data
        _required_columns(df)
        _valid_types(df)
        _nonempty(df)

    def test_parse_row_count(self, autolab_csv_file: Path):
        df = AutolabParser().parse(autolab_csv_file).data
        assert len(df) == 3

    def test_validate_passes(self, autolab_csv_file: Path):
        result = AutolabParser().parse(autolab_csv_file)
        result.validate()


# ---------------------------------------------------------------------------
# Zahner tests
# ---------------------------------------------------------------------------


class TestZahnerParser:
    def test_can_parse_ism(self, zahner_ism_file: Path):
        assert ZahnerParser.can_parse(zahner_ism_file) is True

    def test_cannot_parse_mpt(self, biologic_mpt_file: Path):
        assert ZahnerParser.can_parse(biologic_mpt_file) is False

    def test_parse_returns_parsed_eis(self, zahner_ism_file: Path):
        result = ZahnerParser().parse(zahner_ism_file)
        assert isinstance(result, ParsedEIS)
        assert "Zahner" in result.instrument

    def test_parse_dataframe_columns(self, zahner_ism_file: Path):
        df = ZahnerParser().parse(zahner_ism_file).data
        _required_columns(df)
        _valid_types(df)
        _nonempty(df)

    def test_parse_row_count(self, zahner_ism_file: Path):
        df = ZahnerParser().parse(zahner_ism_file).data
        assert len(df) == 3

    def test_zimag_negative_convention(self, zahner_ism_file: Path):
        df = ZahnerParser().parse(zahner_ism_file).data
        assert (df["zimag"] <= 0).all()

    def test_validate_passes(self, zahner_ism_file: Path):
        result = ZahnerParser().parse(zahner_ism_file)
        result.validate()


# ---------------------------------------------------------------------------
# Auto-detection tests
# ---------------------------------------------------------------------------


class TestDetectParser:
    def test_detect_gamry(self, gamry_dta_file: Path):
        cls = detect_parser(gamry_dta_file)
        assert cls is GamryParser

    def test_detect_biologic_mpt(self, biologic_mpt_file: Path):
        cls = detect_parser(biologic_mpt_file)
        assert cls is BioLogicParser

    def test_detect_zahner_ism(self, zahner_ism_file: Path):
        cls = detect_parser(zahner_ism_file)
        assert cls is ZahnerParser

    def test_detect_autolab_csv(self, autolab_csv_file: Path):
        cls = detect_parser(autolab_csv_file)
        assert cls is AutolabParser

    def test_detect_returns_none_for_unknown_extension(self, tmp_path: Path):
        f = tmp_path / "mystery.xyz"
        f.write_text("gibberish content")
        cls = detect_parser(f)
        # No registered parser handles .xyz
        assert cls is None

    def test_parse_eis_file_gamry(self, gamry_dta_file: Path):
        result = parse_eis_file(gamry_dta_file)
        _required_columns(result.data)
        _nonempty(result.data)

    def test_parse_eis_file_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_eis_file(tmp_path / "nonexistent.dta")

    def test_all_registered_parsers_are_subclasses(self):
        for cls in REGISTERED_PARSERS:
            assert issubclass(
                cls, PotentiostatParser
            ), f"{cls.__name__} is not a subclass of PotentiostatParser"

    def test_all_registered_parsers_have_extensions(self):
        for cls in REGISTERED_PARSERS:
            assert hasattr(cls, "EXTENSIONS") and len(cls.EXTENSIONS) > 0


# ---------------------------------------------------------------------------
# ParsedEIS dataclass tests
# ---------------------------------------------------------------------------


class TestParsedEIS:
    def _make(self, df: pd.DataFrame) -> ParsedEIS:
        return ParsedEIS(
            data=df,
            source_file="test.dta",
            instrument="Test Instrument",
            extra_meta={},
        )

    def test_validate_passes_with_good_data(self):
        df = pd.DataFrame(
            {
                "frequency": [1000.0, 100.0, 10.0],
                "zreal": [10.0, 12.0, 15.0],
                "zimag": [-3.0, -5.0, -9.0],
            }
        )
        self._make(df).validate()  # must not raise

    def test_validate_raises_on_missing_column(self):
        df = pd.DataFrame({"frequency": [1000.0], "zreal": [10.0]})  # no zimag
        with pytest.raises(ValueError, match="zimag"):
            self._make(df).validate()

    def test_validate_raises_on_empty(self):
        df = pd.DataFrame(columns=["frequency", "zreal", "zimag"])
        with pytest.raises(ValueError, match="empty"):
            self._make(df).validate()
