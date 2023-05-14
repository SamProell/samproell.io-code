"""Compare multiple ECG detectors against reference annotaitons."""
import pathlib
import time

import click
import joblib
import pandas as pd
import wfdb.processing

import utils


def process_record(
    rec: utils.PhysioSignal, detector, widths=(0.05, 0.1, 0.15)
) -> pd.Series:
    """Detect peaks and evaluate results."""
    start = time.perf_counter()
    peaks = detector(rec.signal, rec.fs)
    duration = time.perf_counter() - start
    exec_dur = len(rec.signal) / rec.fs / 3600 / duration  # hours per second

    out = pd.Series([exec_dur], index=["exec_dur"])
    for width in widths:
        comp = wfdb.processing.compare_annotations(
            rec.peaks, peaks, window_width=int(width * rec.fs)
        )
        comp.compare()
        out.loc[f"precision{width}"] = comp.positive_predictivity
        out.loc[f"recall{width}"] = comp.sensitivity
    return out


@click.command()
@click.argument("data_root", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default="results.csv")
@click.option("--jobcache", "-c", default="jobcache")
@click.option("--parser", default="slpdb")
@click.option("--jobs", "-j", default=1)
def main(data_root: str, output: str, parser: str, jobs: int, jobcache: str):
    root_path = pathlib.Path(data_root)
    record_names = (root_path / "RECORDS").read_text().splitlines()
    parse_function = utils.PARSERS[parser]
    memory = joblib.Memory(jobcache, verbose=0)

    @memory.cache
    def processor(path) -> pd.DataFrame:
        import detectors  # pylint: disable=wrong-import-position

        try:
            rec = parse_function(str(path))
            data = {}
            for name, detector in detectors.DETECTORS.items():
                data[name] = process_record(rec, detector, widths=(0.05, 0.1, 0.15))
        except (FileNotFoundError, TypeError) as e:
            print(path, "processing failed", str(e))
            return None
        return pd.DataFrame(data).T

    parallel = joblib.Parallel(n_jobs=jobs, verbose=11)
    dataframes = parallel(
        joblib.delayed(processor)(root_path / name) for name in record_names
    )
    results = pd.concat(dict(zip(record_names, dataframes)))
    results = results.reset_index(names=["record", "method"])
    results.to_csv(output, index=False)

    print(results.head())


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
