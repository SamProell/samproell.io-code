"""Extract a sample from the ASL Alphabet dataset.

Provide a root directory containing separate folders for each class.
One or more non-overlapping subsets of the dataset are extracted (with equal
number of instances for each class). For each subset, a new folder is created
and files are copied. See the README for an example.
"""

import pathlib
import shutil
import typing

import click
import numpy as np
import tqdm


def process_splits(splits: typing.Sequence[str]) -> typing.Dict[str, int]:
    return {s[0]: int(s[1]) for s in map(lambda x: x.split(":"), splits)}


@click.command()
@click.argument(
    "input_root",
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "output_root",
    type=click.Path(
        file_okay=False,
        path_type=pathlib.Path,
    ),
)
@click.option("--split", "splits", multiple=True, type=str, default=["train:100"])
@click.option("--seed", default=None, type=int)
def main(
    input_root: pathlib.Path,
    output_root: pathlib.Path,
    splits: str,
    seed: typing.Union[int, None],
):
    split_sizes = process_splits(splits)
    pbar = tqdm.tqdm(sorted(input_root.iterdir()))
    for labelpath in pbar:
        pbar.set_description(labelpath.name)
        files = sorted(labelpath.iterdir())
        rng = np.random.default_rng(seed=seed)
        indices = rng.permutation(len(files))

        offset = 0
        for name, size in split_sizes.items():
            output_name = output_root.name + "_" + name
            outpath = output_root.with_name(output_name) / labelpath.name
            outpath.mkdir(parents=True, exist_ok=True)
            for i in indices[offset : offset + size]:
                shutil.copy(files[i], outpath / files[i].name)
            offset += size


if __name__ == "__main__":
    main()
