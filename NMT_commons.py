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


def sample_from_probabilities(probabilities, topn, vocabulary_size):
    """Roll the dice to produce a random integer in the [0..vocabulary_size] range,
        according to the provided probabilities. If topn is specified, only the
        topn highest probabilities are taken into account.
        :param probabilities: a list of size vocabulary_size with individual probabilities
        :param topn: the number of highest probabilities to consider. Defaults to all of them.
        :return: a random integer
        """
    probabilities = probabilities[-1, :]  # get the last one in time

    p = np.squeeze(probabilities)  # remove the first dimension
    p[np.argsort(p)[:-topn]] = 0  # leave only the topn , zero otherwise

    p = p / np.sum(p)  # normalize
    return np.random.choice(vocabulary_size, 1, p=p)[0]  # get one sample


def make_zip_results(filename, file_version_index, output_file_name):
    file_version_string = ""
    while file_version_index / 26 > 0:
        file_version_string = chr(ord('A') + file_version_index % 26) + file_version_string
        file_version_index //= 26

    myzipfile = ZipFile('{}{}.zip'.format(filename, file_version_string))
    myzipfile.addDir('log/')
    myzipfile.addDir('checkpoints/')
    # myzipfile.addFile('reverse_dictionary.pkl.gz')
    myzipfile.addFile(output_file_name)
    myzipfile.print_info()
