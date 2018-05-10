import numpy as np

from zipFileHelperClass import ZipFile


def negativeLogProb(predictions, labels):  # [seq_length , batch_size,vocabulary] as labels [seq_length,batch_size]
    """Log-probability of the true labels in a predicted batch."""
    predictions = np.reshape(predictions, [-1, predictions.shape[2]])
    labels = np.reshape(labels, [-1])
    predictions[predictions < 1e-10] = 1e-10  # wont go -infinity
    return np.sum(-np.log2(predictions[np.arange(labels.shape[0]), labels])) / labels.shape[0]  # single value


def perplexity(predictions, labels):
    """perplexity of the model."""
    return 2 ** negativeLogProb(predictions, labels)


def sample_from_probabilities(probabilities, topn, vocabulary_size, is_word_level=False):
    """Roll the dice to produce a random integer in the [0..vocabulary_size] range,
        according to the provided probabilities. If topn is specified, only the
        topn highest probabilities are taken into account.
        :param probabilities: a list of size vocabulary_size with individual probabilities
        :param topn: the number of highest probabilities to consider. Defaults to all of them.
        :return: a random integer
        """
    probabilities = probabilities[-1, :]

    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0  # leave only the topn , zero otherwise
    if is_word_level:  # cut the UNK
        p[0] /= 1000
    p = p / np.sum(p)  # normalize
    return np.random.choice(vocabulary_size, 1, p=p)[0]  # get one sample


def make_zip_results(filename, step, outputFileName):
    myzipfile = ZipFile('{}{}.zip'.format(filename, step))
    myzipfile.addDir('log/')
    myzipfile.addDir('checkpoints/')
    # myzipfile.addFile('reverse_dictionary.pkl.gz')
    myzipfile.addFile(outputFileName)
    myzipfile.print_info()
