# Building an ASL alphabet detector with MediaPipe Model Maker

The code provided here is thoroughly described in the post HERE(!).
Using the MediaPipe model maker, we can customize a hand gesture recognition
model to recognize characters from the ASL fingerspelling alphabet.

The models are trained with data from the
[SigNN Character Database][signn-database],
consider also taking a look at and starring the corresponding
[Github repo](https://github.com/AriAlavi/SigNN).


## Install dependencies
Install dependencies listed in the provided requirements file:
```bash
python -m pip install -r requirements.txt
```

## Get the dataset
Download the dataset through [Kaggle][signn-database]. You need a Kaggle account,
which is free.

## Preparing the dataset
Because processing through the Model Maker is slow, the following script was
used to create non-overlapping subsets (with varying sizes) of the dataset.
For example:
```sh
python generate_data_samples.py \
    ./data/SigNN\ Character\ Database/ ./data/SigNN \
    --split "train100:100" --split "train50:50" \
    --split "test10:10" --split "test:50"
```
You can pass multiple `--split` options. Each should be given in the form
`"<name>:<size>"`. For each split, a corresponding folder is created, which
replicates the structure of the dataset, but only uses the specified number of
instances for each class.
The example above would produce:
```
└── data
    ├── SigNN Character Database  # original dataset (unchanged)
    │   ├── A
    │   ├── ...
    │   └── Y
    │
    ├── SigNN_train100  # subset with 100 images each
    │   ├── A
    │   ├── ...
    │   └── Y
    │
    ├── SigNN_train50  # subset with 50 images each
    │   └── ...
    ├── SigNN_test10   # subset with 10 images each
    │   └── ...
    └── SigNN_test     # subset with 50 images each
        └── ...
```
data/SigNN Character Database



[signn-database]: https://www.kaggle.com/datasets/signnteam/asl-sign-language-pictures-minus-j-z
