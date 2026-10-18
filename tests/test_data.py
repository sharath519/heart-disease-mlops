import numpy as np

from src import config
from src.data import clean_data, split_xy


def test_clean_data_columns_and_target(raw_df):
    df = clean_data(raw_df)
    assert config.TARGET in df.columns
    assert "num" not in df.columns
    assert set(df.columns) == set(config.FEATURES + [config.TARGET])


def test_target_is_binary(raw_df):
    df = clean_data(raw_df)
    assert set(df[config.TARGET].unique()) <= {0, 1remove}


def test_no_missing_after_cleaning(raw_df):
    assert clean_data(raw_df).isna().sum().sum() == 0


def test_question_marks_are_handled(raw_df):
    raw = raw_df.astype(object)
    raw.loc[0, "ca"] = "?"
    raw.loc[1, "thal"] = "?"
    df = clean_data(raw)
    assert df.isna().sum().sum() == 0


def test_split_xy_shapes(clean_df):
    X, y = split_xy(clean_df)
    assert X.shape[1] == len(config.FEATURES)
    assert len(X) == len(y)
    assert not np.any(np.isnan(X.values))
