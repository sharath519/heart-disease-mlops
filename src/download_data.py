"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/.

Usage:  python -m src.download_data
Tries the raw UCI file first and falls back to the `ucimlrepo` package.
"""
import logging
import sys

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_data")


def _from_url() -> pd.DataFrame:
    return pd.read_csv(config.UCI_URL, header=None, names=config.COLUMNS)


def _from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=config.UCI_DATASET_ID)
    df = ds.data.features.copy()
    df["num"] = ds.data.targets["num"].values
    return df[config.COLUMNS]


def download(force: bool = False) -> pd.DataFrame:
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if config.RAW_PATH.exists() and not force:
        log.info("Raw data already present at %s", config.RAW_PATH)
        return pd.read_csv(config.RAW_PATH)

    try:
        log.info("Downloading from %s", config.UCI_URL)
        df = _from_url()
    except Exception as exc:  # network / format problems
        log.warning("Direct download failed (%s); trying ucimlrepo", exc)
        df = _from_ucimlrepo()

    df.to_csv(config.RAW_PATH, index=False)
    log.info("Saved %d rows x %d cols to %s", *df.shape, config.RAW_PATH)
    return df


if __name__ == "__main__":
    try:
        download(force="--force" in sys.argv)
    except Exception:
        log.exception(
            "Could not download. Manual option: get processed.cleveland.data from "
            "https://archive.ics.uci.edu/dataset/45/heart+disease and save it as "
            "data/raw/heart_raw.csv (with a header row: %s)", ",".join(config.COLUMNS))
        sys.exit(1)
