from typing import Optional, Tuple
import dataclasses
import pathlib

import numpy as np
import pandas as pd
import wfdb
from wfdb.io.annotation import ann_label_table, is_qrs

QRS_TABLE = ann_label_table.loc[np.array(is_qrs)[ann_label_table.label_store]]
QRS_CODES = QRS_TABLE.symbol


@dataclasses.dataclass
class PhysioSignal:
    signal: np.ndarray
    peaks: np.ndarray
    fs: int
    name: Optional[str] = None


def get_qrs_indices(ann: wfdb.Annotation) -> np.ndarray:
    """Extract only QRS peak indices from WFDB annotation object."""
    ann_series = pd.Series(ann.symbol, index=ann.sample)
    return ann_series[ann_series.isin(QRS_CODES)].index.to_numpy()


def read_slpdb_record(path: str) -> PhysioSignal:
    """Read the ECG signal and annotations for a single SLPDB record."""
    ecg, fields = wfdb.rdsamp(path, channel_names=["ECG"])
    ann = wfdb.rdann(path, extension="ecg")
    rpeaks = get_qrs_indices(ann)
    return PhysioSignal(
        ecg[:, 0], rpeaks, fields["fs"], name=pathlib.Path(path).name
    )


def read_nsrdb_record(path: str) -> PhysioSignal:
    ecg, fields = wfdb.rdsamp(path, channel_names=["ECG1"])
    ann = wfdb.rdann(path, extension="atr")
    rpeaks = get_qrs_indices(ann)
    return PhysioSignal(
        ecg[:, 0], rpeaks, fields["fs"], name=pathlib.Path(path).name
    )


def read_nstdb_record(path: str) -> PhysioSignal:
    ecg, fields = wfdb.rdsamp(path, channel_names=["MLII"])

    ann = wfdb.rdann(path, extension="atr")
    rpeaks = get_qrs_indices(ann)
    return PhysioSignal(
        ecg[:, 0], rpeaks, fields["fs"], name=pathlib.Path(path).name
    )


PARSERS = {
    "slpdb": read_slpdb_record,
    "nsrdb": read_nsrdb_record,
    "nstdb": read_nstdb_record,
}
