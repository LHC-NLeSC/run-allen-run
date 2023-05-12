import pandas as pd
import pytest

from compare import df_round_trip, process_df


@pytest.fixture(
    params=["handcoded", "ghostbuster"],
)
def runs(request):
    return pd.read_csv(f"data/{request.param}.csv")


def test_df_round_trip():
    expect = pd.Series(["int64", "bool"], index=["a", "b"])
    df = pd.DataFrame({"a": ["1", "2"], "b": ["True", "False"]})
    result = df_round_trip(df)
    assert result.dtypes.equals(expect)


def test_process_df(runs):
    df = process_df(runs)
    assert df["params.block_dim"].dtype == "int64"
    assert not df["params.onnx_input"].str.endswith(".onnx").any()
    assert (
        df["tags.branch"].cat.categories == ["ghostbuster", "baseline", "handcoded"]
    ).all()

    if df["params.sequence"].str.startswith("ghostbuster_").any():
        assert df["params.no_infer"].dtype == "bool"
        assert df["params.use_fp16"].dtype == "bool"
        assert df["params.use_int8"].dtype == "bool"
