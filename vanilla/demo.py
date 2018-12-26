import gzip
import pickle

import numpy as np
import tensorflow as tf

from model_configs import ModelNamedConstant
from model_utils import sample_from_probabilities

constants = ModelNamedConstant()
############################ Configs #################################
interactive = True
is_short_model = False
vocab_chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
reverse_dict = dict(zip(range(3, 3 + len(vocab_chars)), vocab_chars))

with gzip.open("./runs/nmt.params", 'rb') as f:
    configs_dict = pickle.load(f)
    vocabulary_size_encoder = configs_dict['vocabulary_size_decoder']
    vocabulary_size_decoder = configs_dict['vocabulary_size_decoder']

sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph("./runs/checkpoints/nmt_1529680947-85201.meta")
saver.restore(sess, "./runs/checkpoints/nmt_1529680947-85201")  # check point name not a file

# graph tensor names
decoder_init_state_from_encoder = ('train/concat:0', 'train/concat_1:0')
decoder_final_state = ('train/rnn/while/Exit_3:0', 'train/rnn/while/Exit_4:0')

tf_keep_prop = 'keep_prop_tf:0'
encoder_inputs = 'train/encoder_inputs:0'
encoder_inputs_length = 'train/encoder_inputs_length:0'
decoder_inputs = "train/decoder_inputs:0"
decoder_inputs_length = "train/decoder_inputs_length:0"
dec_probabilities = 'train/Reshape_1:0'
while True:
    # generate sequence to be feeded to encoder
    if interactive:
        user_input = input("Enter Statement\n")
        set_chars = set(user_input)
        dict_chars = dict(zip(set_chars, range(3, 3 + len(set_chars))))
        reverse_dict = dict(zip(dict_chars.values(), dict_chars.keys()))  # of course it's assumed to be 1 to 1 mapping

        test_sequence = np.full((len(user_input) + 1, 1), constants.KOMYEOS)  # Don't need to mark end of sequence :3

        for i, ch in enumerate(user_input):
            test_sequence[i][0] = dict_chars[ch]
    else:
        test_sequence = np.random.randint(3, vocabulary_size_encoder, (np.random.randint(15, 30), 1))  # don't take zero this is padding
        test_sequence[-1][0] = constants.KOMYEOS  # end of sequence

    test_sequence_target = np.copy(test_sequence)
    test_sequence_target[:test_sequence.shape[0] - 1] = np.flip(test_sequence, axis=0)[1:test_sequence.shape[0]]

    # method 2
    feed_dict = {tf_keep_prop: 1.0,
                 encoder_inputs: test_sequence,
                 encoder_inputs_length: np.array([test_sequence.shape[0]]),  # take the length as is
                 }

    decoder_current_state_np = sess.run(decoder_init_state_from_encoder, feed_dict=feed_dict)

    for char in test_sequence:
        if char == constants.KOMYEOS:
            print(" <KOMYEOS>", end="")
        else:
            print(reverse_dict[char[0]], end="")

    print(" ==> ", end="")
    # we now have the thought vector of the encoder
    next_feed = np.array([[constants.KOMYSOS]])  # feed start of sequence token

    cnt, j = 0, 0
    # keep feeding the decoder,sample .. in a loop.. updating the step
    while j in range(test_sequence.shape[0] + 5) and next_feed[0, 0] != constants.KOMYEOS:

        feed_dict = {tf_keep_prop: 1.0,
                     decoder_inputs: next_feed,
                     decoder_inputs_length: np.array([1]),  # a single character
                     }
        # feed tuple state
        for ss, v in enumerate(decoder_init_state_from_encoder):
            feed_dict[v] = decoder_current_state_np[ss]


        decoder_current_state_np, prediction_probabilities = sess.run([decoder_final_state, dec_probabilities], feed_dict=feed_dict)  # wrap around the state and get the probabilities
        sample = sample_from_probabilities(prediction_probabilities, topn=1, vocabulary_size=vocabulary_size_decoder)  # sample from decoder vocabulary

        next_feed = np.array([[sample]])

        if next_feed[0, 0] == constants.KOMYEOS:  # in decoder vocab
            print(" <KOMYEOS>", end="")
        elif next_feed[0, 0] == constants.KOMYSOS:  # in decoder vocab
            print(" <KOMYSOS>", end="")
        elif next_feed[0, 0] == constants.KOMYPAD:  # in decoder vocab
            print(" <KOMYPAD>", end="")
        else:
            try:
                print(reverse_dict[next_feed[0, 0] - 1], end="")  # in encoder vocab

            except KeyError:
                print(" <KOMYSAD>", end="")

        if j < test_sequence.shape[0] and (test_sequence_target[j][0] == sample - 1 or test_sequence_target[j][0] == sample == constants.KOMYEOS):
            cnt += 1

        j += 1

    print("({}/{})".format(cnt, test_sequence.shape[0]), ".")
    print("*" * 50)
