# Template's Basic packages
import fire
from disaster import config  # noqa

# Basic packages
import string
import time
import pandas as pd
import numpy as np
from ast import literal_eval
import json

# ML packages
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Model persistence
import joblib

# NLP packages
import re
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# MISC
import requests
import zipfile

# Global variables
timestr = time.strftime('%Y%m%d%H%M%S')

# Compile regex (Global)

regex = {}

# hashtags
regex['hash'] = re.compile(r'\#(\w+)')

# "tags" with '@'
regex['at'] = re.compile(r'\@(\w+)')

# URLs beginning with 'http://' or 'https://'
regex['url'] = re.compile(r'http.*\:\/\/(\w+\.\w+)')


def split(**kwargs):
    """
    Function that will generate the train/valid/test sets

    Examples
    --------
    $ make slit

    Notes
    -----
    config.data_path: workspace/data

    You should use workspace/data to put data to working on.  Let's say
    you have workspace/data/iris.csv, which you downloaded from:
    https://archive.ics.uci.edu/ml/datasets/iris. You will generate
    the following:

    + workspace/data/test.csv
    + workspace/data/train.csv
    + workspace/data/valid.csv
    + other files

    With these files you can train your model!
    """
    print("==> GENERATING TRAIN / VALID DATASETS")

    # Load data from file
    path = (
        '{0}/nlp-getting-started/train.csv'
        .format(config.data_path)
    )

    df = pd.read_csv(path)

    mask = np.random.rand(len(df)) < 0.75

    train = df[mask]

    valid = df[~mask]

    # Write data to file

    path = ('{0}/train.csv'.format(config.data_path))
    train.to_csv(path, index=False)

    print('    ==> TRAIN: {0}'.format(len(train)))

    path = ('{0}/valid.csv'.format(config.data_path))
    valid.to_csv(path, index=False)

    print('    ==> VALID: {0}'.format(len(valid)))

    # Load data from file
    path = (
        '{0}/nlp-getting-started/test.csv'
        .format(config.data_path)
        )

    test = pd.read_csv(path)

    path = ('{0}/test.csv'.format(config.data_path))
    test.to_csv(path, index=False)

    print('    ==> TEST: {0}'.format(len(valid)))

    pass


def tokenizer(text):
    """
    Util for cleaning and tokenazing text for NLP tasks

    References
    ----------
    https://machinelearningmastery.com/clean-text-machine-learning-python/
    """

    # split into words
    tokens = word_tokenize(text)

    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    return words


def process(**kwargs):
    """
    Pre-process tabular text for NLP tasks

    Examples
    --------
    $ make process

    """
    print("==> PRE-PROCESSING TEXTUAL DATA")

    for s in ['train', 'valid', 'test']:

        # Load data from file
        path = ('{0}/{1}.csv'.format(config.data_path, s))

        df = pd.read_csv(path)

        # Clean and tokenize

        df['tokens'] = df.apply(
            lambda row: tokenizer(row['text']),
            axis=1
            )

        # Extract words based on regex patterns
        # (pre-compiled as global variables)

        for r in ['hash', 'at', 'url']:
            df[r] = df.apply(
                lambda row: re.findall(regex[r], row['text'].lower()),
                axis=1
                )

        # Write data to file
        path = path.replace('.csv', '_processed.csv')

        df.to_csv(path, index=False)

        pass


def features(**kwargs):
    """
    Function that will generate the dataset for your model. It can
    be the target population, training or validation dataset. You can
    do in this step as well do the task of Feature Engineering.

    Examples
    --------
    $ make features

    Notes
    -----
    config.data_path: workspace/data

    You should use workspace/data to put data to working on.  Let's say
    you have workspace/data/iris.csv, which you downloaded from:
    https://archive.ics.uci.edu/ml/datasets/iris. You will generate
    the following:

    + workspace/data/test.csv
    + workspace/data/train.csv
    + workspace/data/validation.csv
    + other files

    With these files you can train your model!
    """
    print("==> FEATURE ENGINEERING")

    try:

        print("    ==> LOOKING FOR GLOVE-6B")

        path = '{0}/glove6B/glove.6B.300d.txt'.format(config.download_path)

        open(path, 'r')

    except FileNotFoundError:

        # DOWNLOAD: glove6B
        print("    ==> DOWNLOADING GLOVE-6B")

        url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        r = requests.get(url, allow_redirects=True)
        path = '{0}/glove6B.zip'.format(config.download_path)
        open(path, 'wb').write(r.content)

        path = '{0}/glove6B/'.format(config.download_path)

        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(path)
        zip_ref.close()

    print("    ==> LOADING GLOVE-6B")

    path = '{0}/glove6B/glove.6B.300d.txt'.format(config.download_path)

    f = open(path, 'r')

    glove = {}

    for line in f:

        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        glove[word] = embedding

    hot = {}

    for s in ['train', 'valid', 'test']:

        # Load data from file

        path = ('{0}/{1}_processed.csv'.format(config.data_path, s))

        df = pd.read_csv(path)

        # -- FEATURE: SENTIMENT SCORE --

        analyzer = SentimentIntensityAnalyzer()

        df['feature_sent_score'] = df.apply(
                lambda row: (
                    analyzer.polarity_scores(row['text'])['compound']
                ),
                axis=1
                )

        # -- FEATURES: WORDS FROM REGEX PATTERNS --

        for r in ['hash', 'at', 'url']:

            # Replace '[]' by '["none_hash"]' or '["none_at"]'
            # to keep track of null categories
            df[r] = df.apply(
                lambda row: row[r].replace('[]', '["none_{0}"]'.format(r)),
                axis=1
                )

            # 'literal_evar' transforms strings into actual arrays
            df[r] = df.apply(
                lambda row: literal_eval(row[r]),
                axis=1
                )

        # Combine arrays in a single array
        df['array'] = df.apply(
            lambda row: list(set(row['hash'] + row['at'] + row['url'])),
            axis=1
            )

        # Explode array
        df_exploded = (
            df.explode('array')[['id', 'array']]
            .drop_duplicates()
            .set_index('id')
            )

        # -- FEATURES: WORD EMBEDDINGS AVERAGES --

        df['embeddings_array'] = df.apply(
                lambda row: (
                    np.average(
                        [
                            glove[s] for s in
                            literal_eval(row['tokens'])
                            if s in glove.keys()
                        ],
                        axis=0
                    )
                ),
                axis=1
            )

        # Drop nan
        df.dropna(axis=0, subset=['embeddings_array'], inplace=True)

        cols = ['feature_glove_{0}'.format(str(s)) for s in range(0, 300)]

        df.set_index('id', inplace=True)

        df[cols] = pd.DataFrame(
            df['embeddings_array'].values.tolist(),
            index=df.index
            )

        # (1) Get one hot encoding of columns 'array'
        # (2) Join 'feature_sent_score'
        # (3) Join 'feature_glove_...'
        # (4) Fill nan's
        # (5) Reset index

        hot[s] = (
            pd.get_dummies(df_exploded['array'])
            .groupby('id')
            .sum()
            .join(
                df['feature_sent_score'],
                how='left'
                )
            .join(
                df[cols],
                how='left'
                )
            .fillna(0)
            .reset_index()
            )

        # If not in 'train', check columns against
        # 'train' and fill missing columns with 0's
        if s != 'train':
            for r in hot['train'].columns:
                try:
                    hot[s][r]
                except KeyError:
                    hot[s][r] = 0

        # Write data to file
        path = path.replace('_processed.csv', '_features.csv')

        hot[s].to_csv(path, index=False)

        pass


def select(**kwargs):
    """
    Feature selection based on training data only

    Parameters
    ----------
    alpha: float
            ElasticNet hyperparameter
            - Constant that multiplies the penalty terms

    l1_ratio: float
                ElasticNet hyperparameter
                - Mixing parameter

    Examples
    --------
    $ make select PARAMS="--alpha=0.5 --l1_ratio=0.5"

    Notes
    -----
    config.models_path: workspace/models
    config.data_path: workspace/data

    As convention you should use workspace/data to read your dataset,
    which was build from generate() step. You should save your model
    binary into workspace/models directory.
    """
    print("==> FEATURE SELECTION (ON TRAINING SET ONLY)")

    # Defaults
    alpha = 0.01
    l1_ratio = 0.05

    # If 'alpha' passed through PARAMS
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']

    # If 'l1_ratio' passed through PARAMS
    if 'l1_ratio' in kwargs.keys():
        l1_ratio = kwargs['l1_ratio']

    # Load data from file

    path = ('{0}/train.csv'.format(config.data_path))

    df_target = pd.read_csv(path, usecols=['id', 'target'])

    df_target.set_index('id', inplace=True)

    # Load data from file

    path = ('{0}/train_features.csv'.format(config.data_path))

    df_feat = pd.read_csv(path)

    df_feat.set_index('id', inplace=True)

    # Fit an ElasticNet with some amount of Lasso
    clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    clf.fit(df_feat, df_target)

    # ElasticNet coefficients
    coefs = np.array(clf.coef_)

    # Features with non-zero ElasticNet coefficients
    cols = np.array(df_feat.columns)

    best_features = cols[np.abs(coefs) > 1.e-12]

    # Document parameter values

    s = 'alpha = {0}'.format(str(alpha))

    best_features = np.insert(best_features, 0, s)

    s = 'l1_ratio = {0}'.format(str(l1_ratio))

    best_features = np.insert(best_features, 0, s)

    # Write best parameter values and best features to txt

    path = (
        '{0}/{1}_feature_selection.txt'
        .format(config.models_path, timestr)
        )

    np.savetxt(path, best_features, delimiter=',', fmt='%s')

    print(
        "    ==> BEST FEATURES WRITTEN TO FILE\n        {0}"
        .format(path)
        )

    print('\n{0}'.format(best_features))

    pass


def recover_fs(kwargs):
    """
    Recover `best_features` from `select` module

    Parameters
    ----------
    filename_fs: str
                Filename where to read list of features
    """

    # Default based on Global Variable `timestr`
    filename_fs = (
        '{0}_feature_selection.txt'
        .format(timestr)
        )

    # If 'filename_fs' passed through PARAMS
    if 'filename_fs' in kwargs.keys():
        filename_fs = (
            kwargs['filename_fs']
        )

    print(
        "    ==> USING FEATURES LIST FROM FILE\n    {0} :"
        .format(filename_fs)
        )

    # Load features from file (skip first 2 rows)

    path = (
        '{0}/{1}'
        .format(
            config.models_path,
            filename_fs
            )
        )

    try:

        best_features = np.loadtxt(
            path,
            delimiter='\n',
            skiprows=2,
            dtype='str'
            )

    except OSError:

        raise Exception('''
            Error: try passing a filename as parameter.\n
            Example: $ make train PARAMS=\"--filename_fs=\'...\'\"
            ''')

    print('\n{0}'.format(best_features))

    return best_features


def train(**kwargs):
    """
    Function that will run your model, be it a NN, Composite indicator
    or a Decision tree, you name it.

    Parameters
    ----------

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
    $ make train PARAMS="--filename_fs='20200310151753_feature_selection.txt'"

    $ make train PARAMS="--n_jobs=2"

    Notes
    -----
    config.models_path: workspace/models
    config.data_path: workspace/data

    As convention you should use workspace/data to read your dataset,
    which was build from generate() step. You should save your model
    binary into workspace/models directory.
    """
    print("==> TRAINING THE MODEL!")

    # Defaults
    n_estimators = 10000
    max_depth = 8
    min_samples_split = 250
    n_jobs = -2
    random_state = 43

    # If 'n_estimators' passed through PARAMS
    if 'n_estimators' in kwargs.keys():
        n_estimators = kwargs['n_estimators']

    # If 'max_depth' passed through PARAMS
    if 'max_depth' in kwargs.keys():
        max_depth = kwargs['max_depth']

    # If 'min_samples_split' passed through PARAMS
    if 'min_samples_split' in kwargs.keys():
        min_samples_split = kwargs['min_samples_split']

    # If 'n_jobs' passed through PARAMS
    if 'n_jobs' in kwargs.keys():
        n_jobs = kwargs['n_jobs']

    # If 'random_state' passed through PARAMS
    if 'random_state' in kwargs.keys():
        random_state = kwargs['random_state']

    # Recover `best_features`
    best_features = recover_fs(kwargs)

    # Load data from file

    path = ('{0}/train.csv'.format(config.data_path))

    df_target = pd.read_csv(path, usecols=['id', 'target'])

    df_target.set_index('id', inplace=True)

    df_target.sort_index(axis=0, inplace=True)

    # Load data from file

    path = ('{0}/train_features.csv'.format(config.data_path))

    df_feat = pd.read_csv(path)

    df_feat.set_index('id', inplace=True)

    df_feat.sort_index(axis=0, inplace=True)

    # Create the model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_jobs=n_jobs,
        random_state=random_state
        )

    # Fit model on training data
    clf.fit(df_feat[best_features], df_target['target'])

    # Save model

    filename = (
        '{0}_model.joblib'
        .format(timestr)
        )

    path = (
        '{0}/{1}'
        .format(
            config.models_path,
            filename
            )
        )

    joblib.dump(clf, path)

    print(
        "    ==> MODEL PICKLED AT\n    {0} :"
        .format(path)
        )

    pass


def metadata(**kwargs):
    """Generate metadata for model governance using testing!

    Notes
    -----
    workspace_path: config.workspace_path

    In this section you should save your performance model,
    like metrics, maybe confusion matrix, source of the data,
    the date of training and other useful stuff.

    You can save like as workspace/performance.json:

    {
       'name': 'My Super Nifty Model',
       'metrics': {
           'accuracy': 0.99,
           'f1': 0.99,
           'recall': 0.99,
        },
       'source': 'https://archive.ics.uci.edu/ml/datasets/iris'
    }

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    print("==> TESTING MODEL PERFORMANCE AND GENERATING METADATA")

    # Load model
    filename = (
        '{0}_model.joblib'
        .format(timestr)
        )

    path = (
        '{0}/{1}'
        .format(
            config.models_path,
            filename
            )
        )

    print(
        "    ==> USING MODEL FROM FILE\n    {0} :"
        .format(filename)
        )

    clf = joblib.load(path)

    # Recover `best_features`
    best_features = recover_fs(kwargs)

    y_true = {}
    y_pred = {}
    proba = {}
    x = {}

    for s in ['train', 'valid']:

        print('\n    ==> {0}'.format(s.upper()))

        # Load data from file

        path = ('{0}/{1}.csv'.format(config.data_path, s))

        y_true[s] = pd.read_csv(path, usecols=['id', 'target'])

        y_true[s].set_index('id', inplace=True)

        y_true[s].sort_index(axis=0, inplace=True)

        # Load data from file

        path = ('{0}/{1}_features.csv'.format(config.data_path, s))

        x[s] = pd.read_csv(path)

        x[s].set_index('id', inplace=True)

        x[s].sort_index(axis=0, inplace=True)

        thr = 0.5

        proba[s] = clf.predict_proba(x[s][best_features])[:, 1]

        y_pred[s] = (proba[s] > thr).astype(int)

        report = classification_report(y_true[s], y_pred[s], output_dict=True)

        print(report)

        filename = (
            '{0}/{1}_metadata_{2}.json'
            .format(config.models_path, timestr, s)
            )

        with open(filename, 'w') as fp:
            json.dump(report, fp)

    pass


def predict(**kwargs):
    """Predict: load the trained model and score input_data

    Notes
    -----
    As convention you should use predict/ directory
    to do experiments, like predict/input.csv.
    """
    print("==> MAKING PREDICTIONS")

    # Load model
    filename = (
        '{0}_model.joblib'
        .format(timestr)
        )

    path = (
        '{0}/{1}'
        .format(
            config.models_path,
            filename
            )
        )

    print(
        "==> USING MODEL FROM FILE\n    {0} :"
        .format(filename)
        )

    clf = joblib.load(path)

    # Recover `best_features`
    best_features = recover_fs(kwargs)

    y_pred = {}
    proba = {}
    x = {}

    # Load data from file

    path = ('{0}/test_features.csv'.format(config.data_path))

    print("    ==> PREDICT DATASET\n    {0}".format(path))

    x = pd.read_csv(path)

    x.set_index('id', inplace=True)

    x.sort_index(axis=0, inplace=True)

    thr = 0.5

    proba = clf.predict_proba(x[best_features])[:, 1]

    y_pred = (proba >= thr).astype(int)

    output = pd.DataFrame(
        {
            'id': x.reset_index()['id'].values,
            'target': y_pred
        }
    )

    path = (
        '{0}/{1}_test_predict.json'
        .format(config.predict_path, timestr)
        )

    output.to_csv(path, index=False)

    pass


def run(**kwargs):
    """
    Run the full pipeline of the model.
    """
    print("Args: {0}".format(kwargs))
    print("Running disaster by felipepenha")
    split(**kwargs)       # generate train/valid/test sets
    process(**kwargs)     # clean text for NLP tasks
    features(**kwargs)    # generate dataset for training
    select(**kwargs)      # feature selection based on training data only
    train(**kwargs)       # training model and save to filesystem
    metadata(**kwargs)    # performance report
    predict(**kwargs)     # predictions on new data


def skip(**kwargs):
    """
    Run pipeline by skipping split/process/features/select

    Notes
    -----
    You must provide the desired value for the kwarg `filename_fs`
    """
    print("Args: {0}".format(kwargs))
    print("Running disaster by felipepenha")
    train(**kwargs)       # training model and save to filesystem
    metadata(**kwargs)    # performance report
    predict(**kwargs)     # predictions on new data


def cli():
    """
    Caller of the fire cli
    """
    return fire.Fire()


if __name__ == '__main__':
    cli()
