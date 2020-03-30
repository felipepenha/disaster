# --- IMPORT PYTHON PACKAGES ---

# Template's Basic packages
import fire
from disaster import config

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
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

# Model persistence
import joblib

# NLP packages
import re
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Custom modules
from disaster.bert_embedding import BertEmbedding
from disaster.glove_embedding import GloveEmbedding

# -- INITIALIZE NP.RANDOM.SEED --

np.random.seed(43)

# -- PRE-COMPILE REGEX (GLOBAL) --

regex = {}

# hashtags
regex['hash'] = re.compile(r'\#(\w+)')

# "tags" with '@'
regex['at'] = re.compile(r'\@(\w+)')

# URLs beginning with 'http://' or 'https://'
regex['url'] = re.compile(r'http.*\:\/\/(\w+\.\w+)')

# --- INITIALIZE GLOBAL `meta` DICT --

meta = {}

# Timestamp for files
meta['timestr'] = time.strftime('%Y%m%d%H%M%S')

# --- DEFINITIONS FOR THE PIPELINE ---


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
    # remove 'http' and 'https'
    blacklist = ['http', 'https']
    words = [
        word for word in stripped
        if (
            word.isalpha()
            & (word not in blacklist)
        )
    ]

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
    Feature Engineering

    Parameters
    ----------
    opt_features: List[str]
                    List of type of features to include
                    Possible values:
                    'regex', 'vaderSentiment', 'glove', 'bert'

    Examples
    --------
    $ make features PARAMS="--opt_features=[regex,bert]"

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

    # Defaults
    meta['opt_features'] = ['regex', 'vaderSentiment', 'glove']

    # If 'opt_features' passed through PARAMS
    if 'opt_features' in kwargs.keys():
        meta['opt_features'] = kwargs['opt_features']

    hot = {}

    for s in ['train', 'valid', 'test']:

        # Load data from file

        path = ('{0}/{1}_processed.csv'.format(config.data_path, s))

        df = pd.read_csv(path)

        # Create index on 'id'
        df.set_index('id', inplace=True)

        # -- FEATURES: WORDS FROM REGEX PATTERNS --

        if 'regex' in meta['opt_features']:

            for r in ['hash', 'at', 'url']:

                # Replace '[]' by '["none_hash"]' or '["none_at"]'
                # to keep track of null categories
                df[r] = df.apply(
                    lambda row: (
                        row[r]
                        .replace(
                            '[]',
                            '["none_{0}"]'
                            .format(r)
                        )
                    ),
                    axis=1
                )

                # 'literal_evar' transforms strings into actual arrays
                df[r] = df.apply(
                    lambda row: literal_eval(row[r]),
                    axis=1
                )

            # Combine arrays in a single array
            df['array'] = df.apply(
                lambda row: (
                    list(
                        set(
                            row['hash'] +
                            row['at'] +
                            row['url']
                        )
                    )
                ),
                axis=1
            )

            # Explode array
            df_exploded = (
                df
                .reset_index()
                .explode('array')[['id', 'array']]
                .drop_duplicates()
                .set_index('id')
            )

            # (1) Get one hot encoding of columns 'array'
            hot[s] = (
                pd.get_dummies(df_exploded['array'])
                .groupby('id')
                .sum()
            )

        else:

            hot[s] = (
                df
                .reset_index()['id']
                .set_index('id')
                .copy(deep=True)
            )

        # -- FEATURE: SENTIMENT SCORE --

        if 'vaderSentiment' in meta['opt_features']:

            analyzer = SentimentIntensityAnalyzer()

            df['feature_vader_sentiment'] = df.apply(
                lambda row: (
                    analyzer.polarity_scores(row['text'])['compound']
                ),
                axis=1
            )

            # (2) Join 'feature_vader_sentiment'
            hot[s] = (
                hot[s]
                .join(
                    df['feature_vader_sentiment'],
                    how='left'
                )
            )

        # -- FEATURES: GLOVE WORD EMBEDDINGS AVERAGES --

        if 'glove' in meta['opt_features']:

            df['glove_array'] = df.apply(
                lambda row: (
                    GloveEmbedding()
                    .embedding(
                        literal_eval(
                            row['tokens']
                        )
                    )
                ),
                axis=1
            )

            # Drop nan
            df.dropna(axis=0, subset=['glove_array'], inplace=True)

            cols_glove = [
                'feature_glove_{0}'.format(str(s))
                for s in range(0, 300)
            ]

            df[cols_glove] = pd.DataFrame(
                df['glove_array'].values.tolist(),
                index=df.index
            )

            # (3) Join 'feature_glove_...'
            hot[s] = (
                hot[s]
                .join(
                    df[cols_glove],
                    how='left'
                )
            )

        # -- FEATURES: BERT SENTENCE EMBEDDINGS AVERAGES --

        if 'bert' in meta['opt_features']:

            df['sentences'] = (
                df['tokens']
                .apply(lambda row: ' '.join(literal_eval(row)))
                .values
            )

            encoding = np.array(
                BertEmbedding(root=config.download_path)
                .embedding(df['sentences'].values)
            )

            features = [
                np.average(list(k[1:]), axis=1).flatten()
                for k in encoding
            ]

            df['bert_array'] = pd.Series(features)

            # Drop nan
            df.dropna(axis=0, subset=['bert_array'], inplace=True)

            cols_bert = [
                'feature_bert_{0}'.format(str(s))
                for s in range(0, 768)
            ]

            df[cols_bert] = pd.DataFrame(
                df['bert_array'].values.tolist(),
                index=df.index
            )

            # (4) Join 'feature_bert_...'
            hot[s] = (
                hot[s]
                .join(
                    df[cols_bert],
                    how='left'
                )
            )

        # (5) Fill nan's
        # (6) Reset index
        hot[s] = (
            hot[s]
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
        path = ('{0}/{1}_features.csv'.format(config.features_path, s))

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
    meta['alpha'] = 1.
    meta['l1_ratio'] = 5.e-4

    # If 'alpha' passed through PARAMS
    if 'alpha' in kwargs.keys():
        meta['alpha'] = kwargs['alpha']

    # If 'l1_ratio' passed through PARAMS
    if 'l1_ratio' in kwargs.keys():
        meta['l1_ratio'] = kwargs['l1_ratio']

    # Load data from file

    path = ('{0}/train.csv'.format(config.data_path))

    df_target = pd.read_csv(path, usecols=['id', 'target'])

    df_target.set_index('id', inplace=True)

    # Load data from file

    path = ('{0}/train_features.csv'.format(config.features_path))

    df_feat = pd.read_csv(path)

    df_feat.set_index('id', inplace=True)

    # Features: regex
    cols_regex = np.array(
        [
            k for k in df_feat.columns
            if not k.startswith('feature_')
        ]
    )

    # Features: other (vaderSentiment, Glove, Bert)
    cols_other = np.array(
        [
            k for k in df_feat.columns
            if k.startswith('feature_')
        ]
    )

    # Fit an ElasticNet with some amount of Lasso
    clf = ElasticNet(
        alpha=meta['alpha'],
        l1_ratio=meta['l1_ratio']
    )

    clf.fit(df_feat[cols_regex], df_target)

    # ElasticNet coefficients
    coefs = np.array(clf.coef_)

    # Features with non-zero ElasticNet coefficients
    best_features = cols_regex[np.abs(coefs) > 1.e-12]

    # Include all 'other' features (vaderSentiment, Glove, Bert)
    best_features = np.append(best_features, cols_other)

    # Document parameter values

    s = 'alpha = {0}'.format(str(meta['alpha']))

    best_features = np.insert(best_features, 0, s)

    s = 'l1_ratio = {0}'.format(str(meta['l1_ratio']))

    best_features = np.insert(best_features, 0, s)

    # Write best parameter values and best features to txt

    path = (
        '{0}/{1}_feature_selection.txt'
        .format(config.models_path, meta['timestr'])
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
                (not necessary when executing $ make run)
    """

    # Default based on Global `meta['timestr']`
    meta['filename_fs'] = (
        '{0}_feature_selection.txt'
        .format(meta['timestr'])
    )

    # If 'filename_fs' passed through PARAMS
    if 'filename_fs' in kwargs.keys():
        meta['filename_fs'] = (
            kwargs['filename_fs']
        )

    print(
        "    ==> USING FEATURES LIST FROM FILE\n    {0} :"
        .format(meta['filename_fs'])
    )

    # Load features from file (skip first 2 rows)

    path = (
        '{0}/{1}'
        .format(
            config.models_path,
            meta['filename_fs']
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
    meta['n_estimators'] = [
        150,
        300,
        450,
        600,
        750,
        900,
        1050,
        1200
    ]

    meta['max_depth'] = [
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12
    ]

    meta['min_samples_split'] = [
        25,
        50,
        75,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500
    ]

    meta['n_jobs'] = -2
    meta['random_state'] = 43
    meta['class_weight'] = 'balanced'

    # If 'n_estimators' passed through PARAMS
    if 'n_estimators' in kwargs.keys():
        meta['n_estimators'] = kwargs['n_estimators']

    # If 'max_depth' passed through PARAMS
    if 'max_depth' in kwargs.keys():
        meta['max_depth'] = kwargs['max_depth']

    # If 'min_samples_split' passed through PARAMS
    if 'min_samples_split' in kwargs.keys():
        meta['min_samples_split'] = kwargs['min_samples_split']

    # If 'n_jobs' passed through PARAMS
    if 'n_jobs' in kwargs.keys():
        meta['n_jobs'] = kwargs['n_jobs']

    # If 'random_state' passed through PARAMS
    if 'random_state' in kwargs.keys():
        meta['random_state'] = kwargs['random_state']

    # Store the grid in a dictionary
    grid = {}

    grid['n_estimators'] = meta['n_estimators']
    grid['max_depth'] = meta['max_depth']
    grid['min_samples_split'] = meta['min_samples_split']

    # Recover `best_features`
    best_features = recover_fs(kwargs)

    # Load data from file

    path = ('{0}/train.csv'.format(config.data_path))

    df_target = pd.read_csv(path, usecols=['id', 'target'])

    df_target.set_index('id', inplace=True)

    df_target.sort_index(axis=0, inplace=True)

    # Load data from file

    path = ('{0}/train_features.csv'.format(config.features_path))

    df_feat = pd.read_csv(path)

    df_feat.set_index('id', inplace=True)

    df_feat.sort_index(axis=0, inplace=True)

    # Create the model
    rf = RandomForestClassifier(
        random_state=meta['random_state'],
        class_weight=meta['class_weight']
    )

    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=grid,
        n_iter=100,
        cv=5,
        verbose=2,
        random_state=meta['random_state'],
        n_jobs=meta['n_jobs']
    )

    # Fit the random search model
    rf_random.fit(df_feat[best_features], df_target['target'])

    # Update `meta` with best hyperparameters
    meta['best_params_'] = rf_random.best_params_

    # Save model

    filename = (
        '{0}_model.joblib'
        .format(meta['timestr'])
    )

    path = (
        '{0}/{1}'
        .format(
            config.models_path,
            filename
        )
    )

    joblib.dump(rf_random, path)

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
    https://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.classification_report.html
    """
    print("==> TESTING MODEL PERFORMANCE AND GENERATING METADATA")

    # Load model
    filename = (
        '{0}_model.joblib'
        .format(meta['timestr'])
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

    for s in ['train', 'valid']:

        print('\n    ==> {0}'.format(s.upper()))

        # Load data from file

        path = ('{0}/{1}.csv'.format(config.data_path, s))

        y_true = pd.read_csv(path, usecols=['id', 'target'])

        y_true.set_index('id', inplace=True)

        y_true.sort_index(axis=0, inplace=True)

        # Load data from file

        path = ('{0}/{1}_features.csv'.format(config.features_path, s))

        x = pd.read_csv(path)

        x.set_index('id', inplace=True)

        x.sort_index(axis=0, inplace=True)

        proba = clf.predict_proba(x[best_features])[:, 1]

        # Performance extracted from the "ROC curve"
        fpr, tpr, thr = (
            metrics.roc_curve(
            y_true=y_true.values,
            y_score=proba,
            pos_label=1,
            drop_intermediate=False
            )
        )

        meta['AUC'] = metrics.auc(fpr, tpr)
        
        diff = np.abs(tpr - fpr)
        
        # Kolmogorov–Smirnov
        meta['KS'] = np.max(diff)
        
        # Numpy index of the maximum separation between TPR and FPR
        ks_idx = np.argmax(diff)
        
        # Update optimum threshold based on KS criterium
        # -- Last updated will be 'valid', to be used later
        meta['optimal_threshold'] = thr[ks_idx]

        # Predicted classes based on 'optimal_threshold'
        y_pred = (proba >= meta['optimal_threshold']).astype(int)

        meta['classification_report'] = (
            metrics.classification_report(
                y_true,
                y_pred,
                output_dict=True
            )
        )

        print(meta)

        filename = (
            '{0}/{1}_metadata_{2}.json'
            .format(config.models_path, meta['timestr'], s)
        )

        # Export to JSON
        with open(filename, 'w') as fp:
            json.dump(meta, fp, indent=4)

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
        .format(meta['timestr'])
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

    for s in ['train', 'valid', 'test']:

        # Load data from file

        path = ('{0}/{1}_features.csv'.format(config.features_path, s))

        print("    ==> PREDICT DATASET\n    {0}".format(path))

        x = pd.read_csv(path)

        x.set_index('id', inplace=True)

        x.sort_index(axis=0, inplace=True)

        proba = clf.predict_proba(x[best_features])[:, 1]

        # Predicted classes based on 'optimal_threshold'
        # Using 'optimal_threshold' from 'valid' set
        y_pred = (proba >= meta['optimal_threshold']).astype(int)

        output = pd.DataFrame(
            {
                'id': x.reset_index()['id'].values,
                'prob_1': proba,
                'class': y_pred
            }
        )

        path = (
            '{0}/{1}_{2}_predict.csv'
            .format(config.predict_path, meta['timestr'], s)
        )

        output.to_csv(path, index=False)

    pass


# --- RUN PIPELINE ---

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
