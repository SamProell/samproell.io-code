import functools

import biosppy
import ecgdetectors
import heartpy
import neurokit2
import numpy as np
import sleepecg
import wfdb


def heartpy_filtered(x, f):
    # https://github.com/paulvangentcom/heartrate_analysis_python/blob/master/examples/2_regular_ECG/Analysing_a_regular_ECG_signal.ipynb
    filtered = heartpy.filter_signal(
        x, cutoff=0.05, sample_rate=f, filtertype="notch"
    )
    wd, _ = heartpy.process(filtered, f)
    return np.array(wd["peaklist"])


def ecgdetectors_engzee(x, f):
    detector = ecgdetectors.Detectors(f)
    return np.array(detector.engzee_detector(x))


DETECTORS = {
    "neurokit2": lambda x, f: neurokit2.ecg_peaks(x, f)[1]["ECG_R_Peaks"],
    "heartpy-filtered": heartpy_filtered,
    "wfdb-xqrs": functools.partial(wfdb.processing.xqrs_detect, verbose=False),
    "biosppy-engzee": lambda x, f: biosppy.signals.ecg.engzee_segmenter(x, f)[
        0
    ],
    "ecgdetectors-engzee": ecgdetectors_engzee,
    "sleepecg": sleepecg.detect_heartbeats,
}
