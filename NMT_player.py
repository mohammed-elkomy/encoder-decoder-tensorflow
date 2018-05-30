import numpy as np
import tensorflow as tf

from NMT_commons import sample_from_probabilities
from configs import vocabulary_size_decoder, vocabulary_size_encoder, KOMYSOS, KOMYEOS, KOMYPAD

############################ Configs #################################
interactive = True
is_short_model = True
vocab_chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
reverse_dict = dict(zip(range(3, 3 + len(vocab_chars)), vocab_chars))
######################################################################

if is_short_model:
    vocabulary_size_encoder = 69
    vocabulary_size_decoder = 70

sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph('./runs/{}'.format("Short-length model/checkpoints/rnn_train_1527004848-124.meta" if is_short_model else "Long-length model/checkpoints/rnn_train_1527472767-18030.meta"))
saver.restore(sess, './runs/{}'.format("Short-length model/checkpoints/rnn_train_1527004848-124" if is_short_model else "Long-length model/checkpoints/rnn_train_1527472767-18030"))
decoder_current_state = ('train/concat:0', 'train/concat_1:0')
while True:
    # generate sequence to be feeded to encoder
    if interactive:
        user_input = input("Enter Statement\n")
        set_chars = set(user_input)
        dict_chars = dict(zip(set_chars, range(3, 3 + len(set_chars))))
        reverse_dict = dict(zip(dict_chars.values(), dict_chars.keys()))  # of course it's assumed to be 1 to 1 mapping

        test_sequence = np.full((len(user_input) + 1, 1), KOMYEOS)  # Don't need to mark end of sequence :3

        for i, ch in enumerate(user_input):
            test_sequence[i][0] = dict_chars[ch]
    else:
        test_sequence = np.random.randint(3, vocabulary_size_encoder, (np.random.randint(15, 30), 1))  # don't take zero this is padding
        test_sequence[-1][0] = KOMYEOS  # end of sequence

    test_sequence_target = np.copy(test_sequence)
    test_sequence_target[:test_sequence.shape[0] - 1] = np.flip(test_sequence, axis=0)[1:test_sequence.shape[0]]

    # method 2
    feed_dict = {'keep_prop_tf:0': 1.0,
                 'train/encoder_inputs:0': test_sequence,
                 'train/encoder_inputs_length:0': np.array([test_sequence.shape[0]]),  # take the length as is
                 }

    decoder_current_state_np = sess.run(decoder_current_state, feed_dict=feed_dict)

    for char in test_sequence:
        if char == KOMYEOS:
            print(" <KOMYEOS>", end="")
        else:
            print(reverse_dict[char[0]], end="")

    print(" ==> ", end="")
    # we now have the thought vector of the encoder
    next_feed = np.array([[KOMYSOS]])  # feed start of sequence token

    cnt, j = 0, 0
    # keep feeding the decoder,sample .. in a loop.. updating the step
    while j in range(test_sequence.shape[0] + 5) and next_feed[0, 0] != KOMYEOS:
        feed_dict = {'keep_prop_tf:0': 1.0,
                     "train/decoder_inputs:0": next_feed,
                     "train/decoder_inputs_length:0": np.array([1]),  # a single character
                     }
        # feed tuple state
        for ss, v in enumerate(decoder_current_state):
            feed_dict[v] = decoder_current_state_np[ss]

        decoder_current_state_np, prediction_probabilities = sess.run([('train/rnn/while/Exit_2:0', 'train/rnn/while/Exit_3:0'), 'train/Reshape_1:0'], feed_dict=feed_dict)  # wrap around the state and get the probabilities
        sample = sample_from_probabilities(prediction_probabilities, topn=1, vocabulary_size=vocabulary_size_decoder)  # sample from decoder vocabulary

        next_feed = np.array([[sample]])

        if next_feed[0, 0] == KOMYEOS:  # in decoder vocab
            print(" <KOMYEOS>", end="")
        elif next_feed[0, 0] == KOMYSOS:  # in decoder vocab
            print(" <KOMYSOS>", end="")
        elif next_feed[0, 0] == KOMYPAD:  # in decoder vocab
            print(" <KOMYPAD>", end="")
        else:
            try:
                print(reverse_dict[next_feed[0, 0] - 1], end="")  # in encoder vocab

            except KeyError:
                print(" <KOMYSAD>", end="")

        if j < test_sequence.shape[0] and (test_sequence_target[j][0] == sample - 1 or test_sequence_target[j][0] == sample == KOMYEOS):
            cnt += 1

        j += 1

    print("({}/{})".format(cnt, test_sequence.shape[0]), ".")
    print("*" * 50)
