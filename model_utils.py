import numpy as np

def negative_log_prob(predictions, labels):  # [seq_length , batch_size,vocabulary] as labels [seq_length,batch_size]
    """Log-probability of the true labels in a predicted batch."""
    predictions = np.reshape(predictions, [-1, predictions.shape[2]])
    labels = np.reshape(labels, [-1])
    predictions[predictions < 1e-10] = 1e-10  # wont go -infinity
    return np.sum(-np.log2(predictions[np.arange(labels.shape[0]), labels])) / labels.shape[0]  # single value


def perplexity(predictions, labels):
    """perplexity of the model."""
    return 2 ** negative_log_prob(predictions, labels)


def sample_from_probabilities(probabilities, topn, vocabulary_size):
    probabilities = probabilities[-1, :]  # get the last one in time

    p = np.squeeze(probabilities)  # remove the first dimension
    p[np.argsort(p)[:-topn]] = 0  # leave only the topn , zero otherwise

    p = p / np.sum(p)  # normalize
    return np.random.choice(vocabulary_size, 1, p=p)[0]  # get one sample
