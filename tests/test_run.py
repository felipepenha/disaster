# Template's Basic packages
from disaster import config

# Basic packages
import numpy as np
from numpy import dot
from numpy.linalg import norm

# Custom modules
from disaster.bert_embedding import BertEmbedding
from disaster.glove_embedding import GloveEmbedding


def cos_sim(a, b):

    return dot(a, b)/(norm(a)*norm(b))


def test_glove():

    similarity = cos_sim(
        (
            GloveEmbedding().embedding(['king'])
            - GloveEmbedding().embedding(['man'])
        ),
        (
            GloveEmbedding().embedding(['queen'])
            - GloveEmbedding().embedding(['woman'])
        )
    )

    assert abs(similarity - 0.7) < 0.1


def test_bert():

    sentences = [
        "hello world",
        "hi world",
    ]

    encoding = np.array(
        BertEmbedding(root=config.download_path)
        .embedding(sentences)
    )

    features = [np.average(list(k[1:]), axis=1).flatten() for k in encoding]

    similarity = cos_sim(features[0][0], features[1][0])

    assert abs(similarity - 0.7) < 0.1
