"""Data loading and cleaning."""
import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Cleveland frame.

    * '?' placeholders -> NaN, everything coerced to numeric
    * missing values: mode for categorical (ca, thal), median for numeric
    * multi-class label `num` (0-4) -> binary `target` (0 = healthy, 1 = disease)
    """
    df = raw.copy()
    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")

    for col in config.CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    for col in config.NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    df[config.CATEGORICAL_FEATURES] = df[config.CATEGORICAL_FEATURES].astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def load_clean(force: bool = False) -> pd.DataFrame:
    """Return the cleaned dataset, downloading and cleaning it if needed."""
    if config.CLEAN_PATH.exists() and not force:
        return pd.read_csv(config.CLEAN_PATH)

    from src.download_data import download

    raw = download()
    clean = clean_data(raw)
    config.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_PATH, index=False)
    log.info("Wrote cleaned data (%d rows) to %s", len(clean), config.CLEAN_PATH)
    return clean


def split_xy(df: pd.DataFrame):
    return df[config.FEATURES], df[config.TARGET]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = load_clean(force=True)
    print(d.shape)
    print(d[config.TARGET].value_counts())
