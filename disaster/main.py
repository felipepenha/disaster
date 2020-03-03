import fire
from disaster import config  # noqa

from nltk.tokenize import word_tokenize
import string
import pandas as pd

def cleaner(text):
    """
        Not originally included in the Neoway template.
        Util for cleaning cunks of text for NLP tasks
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

def clean_text(**kwargs):
    """
        Not originally included in the Neoway template.
        Clean tabular text for NLP tasks
    """

    # load data

    for s in ['train', 'test']:

        path = '{0}/nlp-getting-started/{1}.csv'.format(config.data_path, s)

        df = pd.read_csv(path)

        df['tokens'] = df.apply(lambda row: cleaner(row['text']), axis=1)

        path = path.replace('.csv', '_clean.csv')

        df.to_csv(path, index=False)


def features(**kwargs):
    """Function that will generate the dataset for your model. It can
    be the target population, training or validation dataset. You can
    do in this step as well do the task of Feature Engineering.

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
    print("==> GENERATING DATASETS FOR TRAINING YOUR MODEL")	


def train(**kwargs):
    """Function that will run your model, be it a NN, Composite indicator
    or a Decision tree, you name it.

    NOTE
    ----
    config.models_path: workspace/models
    config.data_path: workspace/data

    As convention you should use workspace/data to read your dataset,
    which was build from generate() step. You should save your model
    binary into workspace/models directory.
    """
    print("==> TRAINING YOUR MODEL!")


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
    """
    print("==> TESTING MODEL PERFORMANCE AND GENERATING METADATA")


def predict(input_data):
    """Predict: load the trained model and score input_data

    NOTE
    ----
    As convention you should use predict/ directory
    to do experiments, like predict/input.csv.
    """
    print("==> PREDICT DATASET {}".format(input_data))


# Run all pipeline sequentially
def run(**kwargs):
    """Run the complete pipeline of the model.
    """
    print("Args: {}".format(kwargs))
    print("Running disaster by felipepenha")
    clean_text(**kwargs)     # clean text for NLP tasks
    features(**kwargs)       # generate dataset for training
    train(**kwargs)          # training model and save to filesystem
    metadata(**kwargs)       # performance report


def cli():
    """Caller of the fire cli"""
    return fire.Fire()


if __name__ == '__main__':
    cli()
