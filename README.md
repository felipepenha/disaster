# [`disaster`]


A reproducible and versioned Natural Language Processing pipeline for classifying tweets as being associated to a disaster or not.


The current best performance is at 78% macro avg f1-score.


## Pipeline

![](/images/disaster_pipeline.png)


## Data


The dataset is composed by 7,613 tweets (textual data only) labeled as `1` (disaster) or `0` (not disaster).


The actual time of the disaster event is irrelevant, e.g. tweets about Hiroshima has been consistently labeled as `1`, e.g.

```
"We are on our way to Hiroshima. Today is the 70th anniversary of the detonation of the atomic bomb."
```


The data was originally obtained from the Kaggle competition [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data).


The original data should be found in the directory 

```
cd workspace/data/nlp-getting-started/
```


Otherwise, do

```
pip install kaggle
cd workspace/data/
kaggle competitions download -c nlp-getting-started
```


Header:

```
id,keyword,location,text,target
```


In this project, we chose to only use `text` to predict the `target`, to avoid depending on keywords that may not be available in the future. Also, we chose not to rely on `location` either, because the data does not seem to be extensive enough (both in time and space) as to avoid bias.


**Warning:** the test set has actually been leaked, thus the Kaggle leaderboard is to be mostly ignored.


## Folder structure


* [analysis](./analysis/): contains jupyter notebooks for experimentation and analysis
* [disaster](./disaster/): main Python package with source of the model
* [tests](./tests/): contains files used for unit tests
* [workspace](./workspace/): were inputs and outputs live

```
.
├── analysis
│   ├── out
│   └── performance.ipynb
├── CONTRIBUTING.md
├── disaster
│   ├── bert_embedding.py
│   ├── config.py
│   ├── glove_embedding.py
│   ├── __init__.py
│   ├── main.py
│   └── __pycache__
├── disaster.egg-info
├── Dockerfile
├── images
├── Makefile
├── README.md
├── requirements.txt
├── setup.py
├── tests
│   ├── conftest.py
│   ├── __init__.py
│   └── test_run.py
└── workspace
    ├── data
    ├── download
    ├── features
    ├── models
    └── predict
```


## Usage

This manual assumes an Ubuntu Linux operating system, or other Debian Linux distribution.


Clone the project:

```
git clone https://github.com/felipepenha/disaster.git
```


Check whether `docker` is installed by running

```
docker run hello-world
```


The expected result is `Hello from Docker!`. Otherwise,

```
sudo apt update
sudo apt upgrade
sudo reboot
sudo apt install docker.io
```


Then, run

```
cd disaster
make check
make run
```


To include `kwargs`:

```
make run PARAMS="--kwarg=[value of kwarg]"
```


The following `kwargs` are available:

```
    Parameters
    ----------
    alpha: float
            ElasticNet hyperparameter
            - Constant that multiplies the penalty terms

    l1_ratio: float
                ElasticNet hyperparameter
                - Mixing parameter

    filename_fs: str
                Filename where to read list of features
                (not necessary when executing $ make run)

    opt_features: List[str]
                    List of type of features to include
                    Possible values:
                    'regex', 'vaderSentiment', 'glove', 'bert'

    n_estimators: int
                    RandomForestClassifier hyperparameter
                    - The number of trees in the forest

    max_depth: int
                RandomForestClassifier hyperparameter
                - The maximum depth of the tree

    min_samples_split: int
                        RandomForestClassifier hyperparameter
                        - The minimum number of samples required to
                            split an internal node

    n_jobs: int
            The number of jobs to run in parallel
            - n_jobs=-1 uses all processor available
                n_jobs=-2 uses all processors but 1


    random_state: int
                    Controls both the randomness of the
                    bootstrapping of the samples used when
                    building trees (if bootstrap=True)
                    and the sampling of the features to
                    consider when looking for the best split
                    at each node (if max_features < n_features)

    Examples
    --------
    make run PARAMS=\
    "--alpha=0.01 --l1_ratio=0.01 \
    --opt_features=[regex,vaderSentiment,glove,bert] \
    --n_estimators=500 \
    --max_depth=10 \
    --min_samples_split=10 0\
    --n_jobs=4 \
    --random_state=1"
```


In case you would like to run the project step by step, run these sequentally

```
make clean
make check
make split
make process
make features
make select
```


The latter will generate a file named `[timestamp]_feature_selection.txt` where `timestr` is of the form `'%Y%m%d%H%M%S'`. Next,

```
make skip PARAMS="--filename_fs='[timestr]_feature_selection.txt'"
```


At this point, you can change the hyperparameters with kwargs passed to `PARAMS` for further tunning the model beyond the defaults.


Finally, to release a new version:

```
git add .
git commit -m "..."
make release VERSION=X.Y.Z
```

Check [Travis CI](https://travis-ci.org/) to monitor Continuous Integration.

## Metadata

Experiments metadata are found at 

* `/workspace/models/[timestamp]_metadata_train.json`
* `/workspace/models/[timestamp]_metadata_valid.json`

For example, the best performance obtained so far (on the validation set) has been logged as

```
{
    "timestr": "20200319041907",
    "opt_features": [
        "regex",
        "vaderSentiment",
        "glove"
    ],
    "alpha": 1.0,
    "l1_ratio": 0.0005,
    "n_estimators": 1000,
    "max_depth": 8,
    "min_samples_split": 250,
    "n_jobs": -2,
    "random_state": 43,
    "class_weight": "balanced",
    "filename_fs": "20200319041907_feature_selection.txt",
    "classification_report": {
        "0": {
            "precision": 0.8087035358114234,
            "recall": 0.7992831541218638,
            "f1-score": 0.80396575033799,
            "support": 1116
        },
        "1": {
            "precision": 0.7304452466907341,
            "recall": 0.7420537897310513,
            "f1-score": 0.7362037598544572,
            "support": 818
        },
        "accuracy": 0.7750775594622544,
        "macro avg": {
            "precision": 0.7695743912510787,
            "recall": 0.7706684719264576,
            "f1-score": 0.7700847550962235,
            "support": 1934
        },
        "weighted avg": {
            "precision": 0.7756035976000872,
            "recall": 0.7750775594622544,
            "f1-score": 0.7753053014157925,
            "support": 1934
        }
    }
}
```