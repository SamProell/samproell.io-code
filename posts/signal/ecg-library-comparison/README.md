# A simple ECG library benchmark

The code provided here allows you to reproduce the ECG library benchmark
described
[here](https://www.samproell.io/posts/signal/ecg-library-comparison/).

## Setup

1. download `slpdb` dataset from
   [PhysioNet](https://physionet.org/content/slpdb/1.0.0/)
1. install requirements: `pip install -r requirements.txt`

## Perform benchmark
Run the benchmark with
```bash
python benchmark.py path/to/slpdb/ --output results.csv --jobs 4
```

## Extending the benchmark
By default, the benchmark expects data from the
[MIT-BIH Polysomnographic Database](https://physionet.org/content/slpdb/1.0.0/) [^slpdb-ref].
Other PhysioNet datasets can be used by specifying a different "parser".
Parsers for the [nsrdb](https://physionet.org/content/nsrdb/1.0.0/) and
[nstdb](https://physionet.org/content/nstdb/1.0.0/) datasets are provided in
`utils.py`. Run the benchmark with

```bash
python benchmark.py path/to/dataset --parser nstdb
```

You can extend the `PARSERS` dictionary in `utils.py` for additional datasets.


[^slpdb-ref]: Y. Ichimaru and G. B. Moody, “Development of the
    polysomnographic database on CD-ROM,” *Psychiatry and Clinical
    Neurosciences*, vol. 53, no. 2, pp. 175–177, 1999,
    doi:[10.1046/j.1440-1819.1999.00527.x](https://doi.org/10.1046/j.1440-1819.1999.00527.x).
