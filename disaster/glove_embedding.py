# Template's Basic packages
from disaster import config  # noqa

# Basic packages
import numpy as np

# MISC
import requests
import zipfile

"""
Load 300-dim vectors from 'glove.6B.300d.txt'

Returns
-------
dict(List[float])
"""

try:

    print("    ==> LOOKING FOR GLOVE-6B")

    path = '{0}/glove6B.zip'.format(config.download_path)

    open(path, 'r')

    path = '{0}/glove.6B.300d.txt'.format(config.download_path)

    open(path, 'r')

except FileNotFoundError:

    # DOWNLOAD: glove6B
    print(
        "    ==> DOWNLOADING GLOVE-6B TO:\n    {0}/glove.6B.zip"
        .format(config.download_path)
    )

    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    r = requests.get(url, allow_redirects=True)
    path = '{0}/glove6B.zip'.format(config.download_path)
    open(path, 'wb').write(r.content)

    zip_ref = zipfile.ZipFile(path, 'r')
    path = config.download_path
    zip_ref.extractall(path)
    zip_ref.close()

print("    ==> LOADING GLOVE-6B")

path = '{0}/glove.6B.300d.txt'.format(config.download_path)

f = open(path, 'r')

glove = {}

for line in f:

    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    glove[word] = embedding


class GloveEmbedding:
    """
    Glove 6B word embeddings in 300 dimensions
    """

    def embedding(self, tokens):

        return (
            np.average(
                [
                    glove[s] for s in tokens
                    if s in glove.keys()
                ],
                axis=0
            )
        )
