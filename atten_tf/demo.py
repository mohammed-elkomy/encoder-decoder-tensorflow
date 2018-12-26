import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model_configs import ModelNamedConstant
from model_utils import sample_from_probabilities

constants = ModelNamedConstant()


def attention_map_plot(name, attention_mat, source_sent, dist_sent):
    fig = plt.figure(name, figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    # https://matplotlib.org/examples/color/colormaps_reference.html
    cax = ax.matshow(attention_mat, cmap='Purples')
    fig.colorbar(cax)

    fontdict = {'fontsize': 7}

    ax.set_xticklabels([''] + source_sent, fontdict=fontdict, rotation=45)
    ax.set_yticklabels([''] + dist_sent, fontdict=fontdict)


############################ Configs #################################
inverting = False

interactive = True
is_short_model = False
vocab_chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
reverse_dict = dict(zip(range(3, 3 + len(vocab_chars)), vocab_chars))

with gzip.open("./runs/inverting/nmt_atten.params" if inverting else "./runs/non-inverting/nmt_atten_non_inverting.params", 'rb') as f:
    configs_dict = pickle.load(f)
    vocabulary_size_encoder = configs_dict['vocabulary_size_decoder']
    vocabulary_size_decoder = configs_dict['vocabulary_size_decoder']

sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph(
    "./runs/inverting/checkpoints/nmt_atten_1530312220-42566.meta" if inverting else "./runs/non-inverting/checkpoints/nmt_atten_non_inverting_1530544227-58204.meta"
)
saver.restore(sess,
              "./runs/inverting/checkpoints/nmt_atten_1530312220-42566" if inverting else "./runs/non-inverting/checkpoints/nmt_atten_non_inverting_1530544227-58204"
              )  # check point name not a file

# graph tensor names
keep_prop_tf = 'keep_prop_tf:0'
encoder_inputs = 'train/encoder_inputs:0'
encoder_inputs_length = 'train/encoder_inputs_length:0'

decoder_inputs = 'train/decoder_inputs:0'
decoder_inputs_length = 'train/decoder_inputs_length:0'

memory_fw = 'train/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3:0'
memory_bw = 'train/ReverseSequence:0'

decoder_init_state_from_encoder = ('train/concat:0', 'train/concat_1:0')
decoder_final_state = ('train/rnn/while/Exit_4:0', 'train/rnn/while/Exit_5:0')
attention_weights = "train/TensorArrayStack_1/TensorArrayGatherV3:0"  # [seq_len_decoder_decoder, seq_len_encoder(memory), batch]

dec_probabilities = 'train/Reshape_1:0'
while True:
    # generate sequence to be feeded to encoder
    if interactive:
        source_sequence = input("Enter Statement\n")
        set_chars = set(source_sequence)
        dict_chars = dict(zip(set_chars, range(3, 3 + len(set_chars))))
        reverse_dict = dict(zip(dict_chars.values(), dict_chars.keys()))  # of course it's assumed to be 1 to 1 mapping

        source_sequence_np = np.full((len(source_sequence) + 1, 1), constants.KOMYEOS)  # Don't need to mark end of sequence :3

        source_sequence = list(source_sequence)
        for i, ch in enumerate(source_sequence):
            source_sequence_np[i][0] = dict_chars[ch]
        source_sequence.append("<KOMYEOS>")

    else:
        source_sequence_np = np.random.randint(3, vocabulary_size_encoder, (np.random.randint(15, 30), 1))  # don't take zero this is padding
        source_sequence_np[-1][0] = constants.KOMYEOS  # end of sequence

    test_sequence_target = np.copy(source_sequence_np)
    if inverting:
        test_sequence_target[:source_sequence_np.shape[0] - 1] = np.flip(source_sequence_np, axis=0)[1:source_sequence_np.shape[0]]

    # method 2
    feed_dict = {keep_prop_tf: 1.0,
                 encoder_inputs: source_sequence_np,
                 encoder_inputs_length: np.array([source_sequence_np.shape[0]]),  # take the length as is
                 }

    decoder_current_state_np, memory_fw_np, memory_bw_np = sess.run([decoder_init_state_from_encoder, memory_fw, memory_bw], feed_dict=feed_dict)

    for char in source_sequence_np:
        if char == constants.KOMYEOS:
            print(" <KOMYEOS>", end="")
        else:
            print(reverse_dict[char[0]], end="")

    print(" ==> ", end="")
    # we now have the thought vector of the encoder
    next_feed = np.array([[constants.KOMYSOS]])  # feed start of sequence token

    attention_mat_fw = np.zeros((1, source_sequence_np.shape[0]))
    attention_mat_bw = np.zeros((1, source_sequence_np.shape[0]))
    predicted_sentence = []
    cnt, j = 0, 0
    # keep feeding the decoder,sample .. in a loop.. updating the step
    while j in range(source_sequence_np.shape[0] + 5) and next_feed[0, 0] != constants.KOMYEOS:
        feed_dict = {keep_prop_tf: 1.0,
                     decoder_inputs: next_feed,
                     decoder_inputs_length: np.array([1]),  # a single character
                     memory_fw: memory_fw_np,
                     memory_bw: memory_bw_np,
                     }
        # feed tuple state
        for index, tensor_name in enumerate(decoder_init_state_from_encoder):
            feed_dict[tensor_name] = decoder_current_state_np[index]

        decoder_current_state_np, prediction_probabilities, attention_weights_np = sess.run([decoder_final_state, dec_probabilities, attention_weights], feed_dict=feed_dict)  # wrap around the state and get the probabilities
        sample = sample_from_probabilities(prediction_probabilities, topn=1, vocabulary_size=vocabulary_size_decoder)  # sample from decoder vocabulary
        attention_mat_fw = np.row_stack([attention_mat_fw, attention_weights_np[0, 0, 0, :]])
        attention_mat_bw = np.row_stack([attention_mat_bw, attention_weights_np[0, 1, 0, :]])

        # attentions.append(attention_weights_np)

        next_feed = np.array([[sample]])

        if next_feed[0, 0] == constants.KOMYEOS:  # in decoder vocab
            print(" <KOMYEOS>", end="")
            predicted_sentence.append("<KOMYEOS>")
        elif next_feed[0, 0] == constants.KOMYSOS:  # in decoder vocab
            print(" <KOMYSOS>", end="")
            predicted_sentence.append("<KOMYSOS>")
        elif next_feed[0, 0] == constants.KOMYPAD:  # in decoder vocab
            print(" <KOMYPAD>", end="")
            predicted_sentence.append("<KOMYPAD>")
        else:
            try:
                print(reverse_dict[next_feed[0, 0] - 1], end="")  # in encoder vocab
                predicted_sentence.append(reverse_dict[next_feed[0, 0] - 1])
            except KeyError:
                print(" <KOMYSAD>", end="")
                predicted_sentence.append("<KOMYSAD>")

        if j < source_sequence_np.shape[0] and (test_sequence_target[j][0] == sample - 1 or test_sequence_target[j][0] == sample == constants.KOMYEOS):
            cnt += 1

        j += 1

    print("({}/{})".format(cnt, source_sequence_np.shape[0]), ".")
    print("*" * 50)

    if interactive:
        # noinspection PyUnboundLocalVariable
        attention_map_plot("Forward", attention_mat_fw[1:, :], source_sequence, predicted_sentence)
        attention_map_plot("Backward", attention_mat_bw[1:, :], source_sequence[::-1], predicted_sentence)
        plt.show()
