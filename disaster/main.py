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
import joblib

# NLP packages
import re
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

    USAGE
    ----
    $ make slit

    NOTE
    ----
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

    path = ('{0}/valid.csv'.format(config.data_path))
    valid.to_csv(path, index=False)

    # Load data from file
    path = (
        '{0}/nlp-getting-started/test.csv'
        .format(config.data_path)
        )

    test = pd.read_csv(path)

    path = ('{0}/test.csv'.format(config.data_path))
    test.to_csv(path, index=False)

    pass


def tokenizer(text):
    """
    Util for cleaning and tokenazing text for NLP tasks

    REFERENCES
    ----
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

    USAGE
    ----
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

    USAGE
    ----
    $ make features

    NOTE
    ----
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

    hot = {}

    for s in ['train', 'valid', 'test']:

        # Load data from file

        path = ('{0}/{1}_processed.csv'.format(config.data_path, s))

        df = pd.read_csv(path)

        # FEATURE: SENTIMENT SCORE

        analyzer = SentimentIntensityAnalyzer()

        df['feature_sent_score'] = df.apply(
                lambda row: (
                    analyzer.polarity_scores(row['text'])['compound']
                ),
                axis=1
                )

        # FEATURES: WORDS FROM REGEX PATTERNS

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

        df.set_index('id', inplace=True)

        # Get one hot encoding of columns 'array'

        hot[s] = (
            pd.get_dummies(df_exploded['array'])
            .groupby('id')
            .sum()
            .join(df['feature_sent_score'])
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

    USAGE
    ----
    $ make select PARAMS="--alpha=... --l1_ratio=..."

    NOTE
    ----
    config.models_path: workspace/models
    config.data_path: workspace/data

    As convention you should use workspace/data to read your dataset,
    which was build from generate() step. You should save your model
    binary into workspace/models directory.
    """
    print("==> FEATURE SELECTION (ON TRAINING SET ONLY)")

    # Defaults
    alpha = 0.01
    l1_ratio = 0.02

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

    best_features = cols[np.abs(coefs) > 1.e-3]

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
        "==> BEST FEATURES WRITTEN TO FILE\n    {0}"
        .format(path)
        )

    print('\n{0}'.format(best_features))

    pass


def train(**kwargs):
    """
    Function that will run your model, be it a NN, Composite indicator
    or a Decision tree, you name it.

    USAGE
    ----
    $ make train PARAMS="--filename_fs='...'"

    'filename_fs': where best features have been stored

    NOTE
    ----
    config.models_path: workspace/models
    config.data_path: workspace/data

    As convention you should use workspace/data to read your dataset,
    which was build from generate() step. You should save your model
    binary into workspace/models directory.
    """
    print("==> TRAINING THE MODEL!")

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
        "==> USING FEATURES LIST FROM FILE\n    {0} :"
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

    # Create the model with 100 trees
    clf = RandomForestClassifier(
        max_depth=10,
        min_samples_split=50,
        random_state=43
        )

    # Fit on training data
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
        "==> MODEL PICKLED AT\n    {0} :"
        .format(path)
        )

    pass


def metadata(**kwargs):
    """Generate metadata for model governance using testing!

    NOTE
    ----
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

    REFERENCES
    ----
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
        "==> USING MODEL FROM FILE\n    {0} :"
        .format(filename)
        )

    clf = joblib.load(path)

    # Default based on Global Variable `timestr`
    filename_fs = (
        '{0}_feature_selection.txt'
        .format(timestr)
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
            Example: $ make metadata PARAMS=\"--filename_fs=\'...\'\"
            ''')

    print('\n{0}'.format(best_features))

    y_true = {}
    y_pred = {}
    proba = {}
    x = {}

    for s in ['train', 'valid']:

        print('\n==> {0}'.format(s.upper()))

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

# def predict(input_data):
#     """Predict: load the trained model and score input_data

#     NOTE
#     ----
#     As convention you should use predict/ directory
#     to do experiments, like predict/input.csv.
#     """
#     print("==> PREDICT DATASET {}".format(input_data))


# Run all pipeline sequentially
def run(**kwargs):
    """Run the complete pipeline of the model.
    """
    print("Args: {}".format(kwargs))
    print("Running disaster by felipepenha")
    split(**kwargs)       # generate train/valid/test sets
    process(**kwargs)     # clean text for NLP tasks
    features(**kwargs)    # generate dataset for training
    select(**kwargs)      # feature selection based on training data only
    train(**kwargs)       # training model and save to filesystem
    metadata(**kwargs)    # performance report


def cli():
    """Caller of the fire cli"""
    return fire.Fire()


if __name__ == '__main__':
    cli()
