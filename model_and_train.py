import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from Logger_Helper import training_summary_log, validation_log, file_saved_log
from NMT_commons import sample_from_probabilities, perplexity, negativeLogProb, make_zip_results
from configs import vocabulary_size_decoder, vocabulary_size_encoder, max_seq_len_encoder, num_samples_train, num_buckets_test, num_buckets_train, encoder_embedding_size, stacked_layers, internal_state_encoder, use_embedding, batch_size, keep_prop, num_epochs, decoder_embedding_size, internal_state_decoder, learning_rate, num_samples_test, KOMYSOS, summary_every_mins, minute, outputFileName, chechpoint_every_mins, max_seq_len_decoder, KOMYEOS, KOMYPAD, is_local_env, print_init_configs
from data_generator import DataGenerator

print_init_configs()

graph = tf.Graph()
with graph.as_default():
    # shared between train and test
    keep_prop_tf = tf.placeholder(dtype=tf.float32, name="keep_prop_tf")
    # repeat it stacked_layers
    encoder_dropcells_fw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(internal_state_encoder), output_keep_prob=keep_prop_tf) for _ in range(stacked_layers)]
    encoder_multi_cell_fw = rnn.MultiRNNCell(encoder_dropcells_fw, state_is_tuple=True)

    encoder_dropcells_bw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(internal_state_encoder), output_keep_prob=keep_prop_tf) for _ in range(stacked_layers)]
    encoder_multi_cell_bw = rnn.MultiRNNCell(encoder_dropcells_bw, state_is_tuple=True)

    decoder_dropcells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(internal_state_decoder), output_keep_prob=keep_prop_tf) for _ in range(stacked_layers)]
    decoder_multi_cell = rnn.MultiRNNCell(decoder_dropcells, state_is_tuple=True)

    with tf.variable_scope('train'):
        # input placeholders
        encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')  # [en_seq_len, batch  ]
        encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')  # [batch]

        decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')  # [de_seq_len, batch] starts with KOKO_START token
        decoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_inputs_length')  # [batch]  IMPORTANT NOTE : decoder_inputs_length = counts the start token
        decoder_outputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_outputs')  # [de_seq_len, batch ] = ground truth padded at the end with zeros

        # embedding lookup or one hot encoding of my languages (encoder / decoder)
        if use_embedding:
            encoder_embeddings = tf.Variable(tf.random_uniform([vocabulary_size_encoder, encoder_embedding_size], -1.0, 1.0), dtype=tf.float32)
            encoder_inputs_to_rnn = tf.nn.embedding_lookup(encoder_embeddings, encoder_inputs)  # [  sequence_length,batch_size, encoder_embedding_size ] # embedded

            decoder_embeddings = tf.Variable(tf.random_uniform([vocabulary_size_decoder, decoder_embedding_size], -1.0, 1.0), dtype=tf.float32)
            decoder_inputs_to_rnn = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)  # [  sequence_length,batch_size, decoder_embedding_size ] # embedded

        else:
            encoder_inputs_to_rnn = tf.one_hot(encoder_inputs, vocabulary_size_encoder, 1.0, 0.0)  # [  sequence_length,batch_size, vocabulary_size ]  # one hot encoded
            decoder_inputs_to_rnn = tf.one_hot(decoder_inputs, vocabulary_size_decoder, 1.0, 0.0)  # [  sequence_length,batch_size, vocabulary_size ]  # one hot encoded

        # encoder reader
        ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = \
            (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_multi_cell_fw, cell_bw=encoder_multi_cell_bw, inputs=encoder_inputs_to_rnn, sequence_length=encoder_inputs_length, dtype=tf.float32, time_major=True))
        # outputs :[sequence_length, batch_size, internal_state_encoder]
        # final_state:[batch_size, internal_state_encoder]

        # this is my thought vector = [batch_size, internal_state_encoder]
        # i will feed this thought vector at inference
        decoder_current_state = tuple([tf.concat((encoder_fw_final_state[i], encoder_bw_final_state[i]), axis=1) for i in range(stacked_layers)])  # state is tuple for decoder input

        # decoder dynamic rnn
        decoder_states_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_multi_cell, inputs=decoder_inputs_to_rnn, initial_state=decoder_current_state, time_major=True, sequence_length=decoder_inputs_length)
        # decoder_states_outputs :[sequence_length, batch_size, internal_state_decoder]
        # decoder_final_state :[batch_size, internal_state_decoder] as tuple repeated stack times

        dec_logits = tf.layers.dense(decoder_states_outputs, units=vocabulary_size_decoder, use_bias=True)  # projection on the vocabulary  outputs : [sequence_length, batch_size, vocabulary_size_decoder]
        dec_probabilities = tf.nn.softmax(dec_logits)  # [sequence_length, batch_size, vocabulary_size_decoder]

        # the triangle has the decoder shape not the encoder !!!!
        lower_triangular_ones = tf.constant(np.tril(np.ones([max_seq_len_decoder, max_seq_len_decoder])), dtype=tf.float32)  # lower triangle ones [max_seq_len_encoder,max_seq_len_encoder] >> [[1. 0.],[1. 1.]]
        _, batch_size_tf = tf.unstack(tf.shape(encoder_inputs))  # seq_length , batch_size

        seqlen_mask = tf.transpose(tf.slice(tf.gather(lower_triangular_ones, decoder_inputs_length - 1), begin=[0, 0], size=[batch_size_tf, tf.reduce_max(decoder_inputs_length)]))  # so you need to take length -1 due to lower triangle ones [sequence_length, batch_size]

        # connect outputs to
        with tf.name_scope("optimization"):
            # Loss function
            loss = tf.contrib.seq2seq.sequence_loss(dec_logits, decoder_outputs, seqlen_mask)  # sparse softmax cross entropy
            # Optimizer
            train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        # To calculate the number correct, this means we don't count the padded as correct
        correct = tf.cast(tf.equal(tf.cast(tf.argmax(dec_logits, 2), tf.int32), decoder_outputs), dtype=tf.float32) * seqlen_mask
        accuracy = tf.reduce_sum(correct) / tf.reduce_sum(seqlen_mask)

        # summary tensors
        if not is_local_env:
            loss_summary = tf.summary.scalar("batch_loss", loss)
            acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
            summaries = tf.summary.merge([loss_summary, acc_summary])

    if not is_local_env:
        saver = tf.train.Saver(max_to_keep=1)

# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'.

if not is_local_env:
    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
    validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not is_local_env:
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

dataGenerator = DataGenerator(num_epochs=num_epochs, vocab=vocabulary_size_encoder, max_len=max_seq_len_encoder, samples=num_samples_train, buckets=num_buckets_train, batch_size=batch_size, special_chars=2)
dataGenerator.average_padding()  # evaluating bucketed padding

execution_start = time.time()

checkpoint_last = time.time()
summary_log_last = time.time()
summary_global_step = 0
zip_file_version = 0
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    step = 0
    tr_accuracy = 0
    ts_accuracy = 0
    for is_end_of_epoch, encoder_in, decoder_in, decoder_out, encoder_len, decoder_len in dataGenerator.next_batch(batch_size):
        current_time = time.time()

        step += 1
        summary_global_step += 1

        feed_dict = {keep_prop_tf: keep_prop,
                     encoder_inputs: encoder_in,
                     encoder_inputs_length: encoder_len,
                     decoder_inputs: decoder_in,
                     decoder_inputs_length: decoder_len,
                     decoder_outputs: decoder_out}

        if (current_time - summary_log_last) > summary_every_mins * minute and step != 0:
            summary_log_last = time.time()
            # print and save some summary

            if not is_local_env:
                prediction_probabilities, smm, accuracy_, loss_ = sess.run([dec_probabilities, summaries, accuracy, loss], feed_dict=feed_dict)
            else:
                prediction_probabilities, accuracy_, loss_ = sess.run([dec_probabilities, accuracy, loss], feed_dict=feed_dict)

            tr_accuracy += accuracy_

            t_perplexity = float(perplexity(prediction_probabilities, decoder_out))
            tr_accuracy /= step

            training_summary_log(summary_global_step, dataGenerator.current_epoch, tr_accuracy, loss_, t_perplexity, execution_start)

            step, tr_accuracy = 0, 0

            # save training data for Tensorboard
            if not is_local_env:
                summary_writer.add_summary(smm, summary_global_step)
                summary_per = tf.Summary(value=[
                    tf.Summary.Value(tag="perplexity", simple_value=t_perplexity),
                ])
                summary_writer.add_summary(summary_per, step)

            print("Example generation")
            for _ in range(5):
                # generate sequence to be feeded to encoder
                test_sequence = np.random.randint(3, vocabulary_size_encoder, (np.random.randint(max_seq_len_encoder//2, max_seq_len_encoder), 1))  # don't take zero this is padding

                test_sequence[-1][0] = KOMYEOS  # end of sequence

                test_sequence_target = np.copy(test_sequence)
                test_sequence_target[:test_sequence.shape[0] - 1] = np.flip(test_sequence, axis=0)[1:test_sequence.shape[0]]

                # # method 1
                #
                # decoder_feed = np.array([[KOMYSOS]])  # feed start of sequence token
                #
                # for char in test_sequence:
                #     if char == KOMYEOS:
                #         print("<KOMYEOS>", end=" ")
                #     else:
                #         print(char[0], end=" ")
                #
                # print(" ==> ", end="")
                #
                #
                # cnt, j = 0, 0
                # # keep feeding the decoder,sample .. in a loop.. updating the step
                # while j in range(test_sequence.shape[0] + 5) and next_feed[0, 0] != KOMYEOS:
                #
                #     feed_dict = {keep_prop_tf: 1.0,
                #                  encoder_inputs: test_sequence,
                #                  encoder_inputs_length: np.array([test_sequence.shape[0]]),  # take the length as is
                #
                #                  decoder_inputs: decoder_feed,
                #                  decoder_inputs_length: np.array([decoder_feed.shape[0]]),  # take the length as is
                #                  }
                #
                #     dec_logits_np, = sess.run([dec_logits], feed_dict=feed_dict)  # incrementally make the input
                #
                #     prediction = dec_logits_np[-1, :].argmax(axis=-1)
                #     decoder_feed = np.concatenate([decoder_feed, prediction[:, None]], axis=0)
                #
                #     if prediction[0] == KOMYEOS:  # in decoder vocab
                #         print("<KOMYEOS>", end=" ")
                #     elif prediction[0] == KOMYSOS:  # in decoder vocab
                #         print("<KOMYSOS>", end=" ")
                #     elif prediction[0] == KOMYPAD:  # in decoder vocab
                #         print("<KOMYPAD>", end=" ")
                #     else:
                #         print(prediction[0] - 1, end=" ")  # in encoder vocab
                #
                #     # do comparison
                #     if j < test_sequence.shape[0] and (test_sequence_target[j][0] == prediction[0] - 1 or test_sequence_target[j][0] == prediction[0] == KOMYEOS):
                #         cnt += 1
                #
                #     j += 1
                #
                # print("({}/{})".format(cnt, test_sequence.shape[0]), ".")

                # method 2
                feed_dict = {keep_prop_tf: 1.0,
                             encoder_inputs: test_sequence,
                             encoder_inputs_length: np.array([test_sequence.shape[0]]),  # take the length as is
                             }

                decoder_current_state_np, = sess.run([decoder_current_state], feed_dict=feed_dict)

                for char in test_sequence:
                    if char == KOMYEOS:
                        print("<KOMYEOS>", end=" ")
                    else:
                        print(char[0], end=" ")

                print(" ==> ", end="")
                # we now have the thought vector of the encoder
                next_feed = np.array([[KOMYSOS]])  # feed start of sequence token

                cnt, j = 0, 0
                # keep feeding the decoder,sample .. in a loop.. updating the step
                while j in range(test_sequence.shape[0] + 5) and next_feed[0, 0] != KOMYEOS:
                    feed_dict = {keep_prop_tf: 1.0,
                                 decoder_inputs: next_feed,
                                 decoder_inputs_length: np.array([1]),  # a single character
                                 }

                    # feed tuple state
                    for ss, v in enumerate(decoder_current_state):
                        feed_dict[v] = decoder_current_state_np[ss]

                    decoder_current_state_np, prediction_probabilities = sess.run([decoder_final_state, dec_probabilities], feed_dict=feed_dict)  # wrap around the state and get the probabilities
                    sample = sample_from_probabilities(prediction_probabilities, topn=1 if dataGenerator.current_epoch <= 1 else 1, vocabulary_size=vocabulary_size_decoder)  # sample from decoder vocabulary

                    next_feed = np.array([[sample]])

                    if next_feed[0, 0] == KOMYEOS:  # in decoder vocab
                        print("<KOMYEOS>", end=" ")
                    elif next_feed[0, 0] == KOMYSOS:  # in decoder vocab
                        print("<KOMYSOS>", end=" ")
                    elif next_feed[0, 0] == KOMYPAD:  # in decoder vocab
                        print("<KOMYPAD>", end=" ")
                    else:
                        print(next_feed[0, 0] - 1, end=" ")  # in encoder vocab

                    if j < test_sequence.shape[0] and (test_sequence_target[j][0] == sample - 1 or test_sequence_target[j][0] == sample == KOMYEOS):
                        cnt += 1

                    j += 1

                print("({}/{})".format(cnt, test_sequence.shape[0]), ".")
                print("*" * 50)
        else:
            _, accuracy_ = sess.run([train_step, accuracy], feed_dict=feed_dict)
            tr_accuracy += accuracy_

        if is_end_of_epoch:
            step = 0
            test_dataGenerator = DataGenerator(num_epochs=1, vocab=vocabulary_size_encoder, max_len=max_seq_len_encoder, samples=num_samples_test, buckets=num_buckets_test, batch_size=batch_size, special_chars=2)

            # eval test set
            te_epoch = test_dataGenerator.current_epoch
            valid_logprob = 0
            valid_loss = 0
            smm = 0
            for t_is_end_of_epoch, t_encoder_in, t_decoder_in, t_decoder_out, t_encoder_len, t_decoder_len in test_dataGenerator.next_batch(batch_size):
                step += 1
                feed_dict = {keep_prop_tf: 1.0,
                             encoder_inputs: t_encoder_in,
                             encoder_inputs_length: t_encoder_len,
                             decoder_inputs: t_decoder_in,
                             decoder_inputs_length: t_decoder_len,
                             decoder_outputs: t_decoder_out}
                loss_, smm, accuracy_, prediction_probabilities = sess.run([loss, summaries, accuracy, dec_probabilities], feed_dict=feed_dict)
                ts_accuracy += accuracy_
                valid_loss += loss_
                valid_logprob = valid_logprob + negativeLogProb(prediction_probabilities, t_decoder_out)

            v_perplexity = float(2 ** (valid_logprob / step))  # our base is 2 we also use log2 at negativeLogProb

            if not is_local_env:
                validation_writer.add_summary(smm, summary_global_step)
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="perplexity", simple_value=v_perplexity),
                ])

                validation_writer.add_summary(summary, summary_global_step)

            valid_loss /= step
            ts_accuracy /= step

            validation_log(step, dataGenerator.current_epoch, ts_accuracy, valid_loss, v_perplexity, execution_start)

            step, ts_accuracy, tr_accuracy = 0, 0, 0

        # save a checkpoint
        if (current_time - checkpoint_last) > chechpoint_every_mins * minute:
            checkpoint_last = time.time()

            if not is_local_env:
                saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=summary_global_step)
                file_saved_log(saved_file, execution_start)
                make_zip_results("NMT", zip_file_version, outputFileName)
                zip_file_version += 1

        sys.stdout.flush()

summary_writer.close()
validation_writer.close()
