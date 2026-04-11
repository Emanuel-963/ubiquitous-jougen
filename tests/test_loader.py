import pathlib
import textwrap

from src.loader import load_eis_file


def write_tmp(path: pathlib.Path, content: str):
    path.write_text(textwrap.dedent(content))


def test_load_csv_commas_and_decimal(tmp_path):
    p = tmp_path / "test1.csv"
    write_tmp(
        p,
        """
    frequency;z';z''
    1;10,0;-1,0
    10;5,0;-0,5
    """,
    )

    df = load_eis_file(str(p))
    assert "frequency" in df.columns
    assert df["frequency"].dtype.kind in "fiu"
    assert df.shape[0] == 2


def test_load_tsv_and_commas(tmp_path):
    p = tmp_path / "test2.tsv"
    write_tmp(
        p,
        """
    freq\tzreal\tzimag
    1\t10,0\t-1,0
    10\t5,0\t-0,5
    """,
    )

    df = load_eis_file(str(p))
    assert df.shape[0] == 2


def test_unparseable_file(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("one two")
    try:
        load_eis_file(str(p))
        assert False, "should raise"
    except ValueError:
        assert True
