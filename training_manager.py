import numpy as np
import tensorflow as tf

from base_TrainingManager import BaseTrainingManager
from data_generator import DataGenerator
from model_configs import ModelConfigs
from model_graph import ModelGraph
from model_utils import perplexity, sample_from_probabilities, negative_log_prob


class TrainingManager(BaseTrainingManager):
    acc_loss: float
    acc_perplexity: float
    acc_accuracy: float

    def __init__(self,
                 configs: ModelConfigs,
                 project_name: str,
                 summary_every_mins: int,
                 checkpoint_every_mins: int,
                 example_every_mins: int,
                 test_every_mins: int,
                 resume_training: bool):

        super(TrainingManager, self).__init__(configs, project_name, summary_every_mins, checkpoint_every_mins, example_every_mins, test_every_mins, resume_training)

    def local_machine_protection(self):  # my machine crashes due to memory limits
        # reduce dataset size
        self.configs.num_samples_train = 256 * 16 * 4
        self.configs.num_samples_test = 256 * 4
        self.configs.max_seq_len_encoder = 30

        self.minute = 0
        self.is_local_env = False

    # to be shared in training loop
    def initialize_accumulated_variables(self):
        self.acc_loss = 0
        self.acc_accuracy = 0
        self.acc_perplexity = 0

    def summary_callback(self, _next, graph: ModelGraph, configs: ModelConfigs, sess: tf.Session):

        encoder_in, decoder_in, decoder_out, encoder_decoder_len = _next

        feed_dict = {graph.keep_prop_tf: configs.keep_prop,
                     graph.encoder_inputs: encoder_in,
                     graph.encoder_inputs_length: encoder_decoder_len,
                     graph.decoder_inputs: decoder_in,
                     graph.decoder_inputs_length: encoder_decoder_len,
                     graph.decoder_outputs: decoder_out}

        # print and save some summary

        if not self.is_local_env:
            # noinspection PyUnboundLocalVariable
            dec_probabilities, summaries, accuracy, loss = sess.run([graph.dec_probabilities, graph.summaries, graph.accuracy, graph.loss], feed_dict=feed_dict)
        else:
            dec_probabilities, accuracy, loss = sess.run([graph.dec_probabilities, graph.accuracy, graph.loss], feed_dict=feed_dict)

        self.acc_accuracy += accuracy

        t_perplexity = float(perplexity(dec_probabilities, decoder_out))

        self.acc_accuracy /= self.step

        self.logger.log_training_summary(self.acc_accuracy, self.acc_loss, self.acc_perplexity)

        # save training data for Tensorboard
        if not self.is_local_env:
            # noinspection PyUnboundLocalVariable
            self.training_writer.add_summary(summaries, self.summary_global_step)

            summary_per = tf.Summary(value=[
                tf.Summary.Value(tag="perplexity", simple_value=t_perplexity),
            ])

            self.training_writer.add_summary(summary_per, self.summary_global_step)

    def sustaining_callback(self, _next, graph: ModelGraph, configs: ModelConfigs, sess: tf.Session):
        encoder_in, decoder_in, decoder_out, encoder_decoder_len = _next
        feed_dict = {graph.keep_prop_tf: configs.keep_prop,
                     graph.encoder_inputs: encoder_in,
                     graph.encoder_inputs_length: encoder_decoder_len,
                     graph.decoder_inputs: decoder_in,
                     graph.decoder_inputs_length: encoder_decoder_len,
                     graph.decoder_outputs: decoder_out}

        _, dec_probabilities, accuracy, loss = sess.run([graph.train_step, graph.dec_probabilities, graph.accuracy, graph.loss], feed_dict=feed_dict)

        self.acc_accuracy += accuracy
        self.acc_perplexity += float(perplexity(dec_probabilities, decoder_out))
        self.acc_loss += loss

    def valid_example_callback(self, graph: ModelGraph, configs: ModelConfigs, sess: tf.Session):
        print("Example generation")
        for _ in range(5):
            # generate sequence to be feeded to encoder
            test_sequence = np.random.randint(3, configs.vocabulary_size_encoder, (np.random.randint(configs.max_seq_len_encoder // 2, configs.max_seq_len_encoder), 1))

            test_sequence[-1][0] = configs.named_constants.KOMYEOS  # end of sequence

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

            feed_dict = {graph.keep_prop_tf: 1.0,
                         graph.encoder_inputs: test_sequence,
                         graph.encoder_inputs_length: np.array([test_sequence.shape[0]]),  # take the length as is
                         }

            decoder_current_state_np, = sess.run([graph.decoder_current_state], feed_dict=feed_dict)

            for char in test_sequence:
                if char == configs.named_constants.KOMYEOS:
                    print("<KOMYEOS>", end=" ")
                else:
                    print(char[0], end=" ")

            print(" ==> ", end="")
            # we now have the thought vector of the encoder
            next_feed = np.array([[configs.named_constants.KOMYSOS]])  # feed start of sequence token

            cnt, j = 0, 0
            # keep feeding the decoder,sample .. in a loop.. updating the step
            while j in range(test_sequence.shape[0] + 5) and next_feed[0, 0] != configs.named_constants.KOMYEOS:
                feed_dict = {graph.keep_prop_tf: 1.0,
                             graph.decoder_inputs: next_feed,
                             graph.decoder_inputs_length: np.array([1]),  # a single character
                             }

                # feed tuple state
                for index, tensor_name in enumerate(graph.decoder_current_state):
                    feed_dict[tensor_name] = decoder_current_state_np[index]

                decoder_current_state_np, prediction_probabilities = sess.run([graph.decoder_final_state, graph.dec_probabilities], feed_dict=feed_dict)  # wrap around the state and get the probabilities
                sample = sample_from_probabilities(prediction_probabilities, topn=1 if self.data_generator.current_epoch <= 1 else 1, vocabulary_size=configs.vocabulary_size_decoder)  # sample from decoder vocabulary

                next_feed = np.array([[sample]])

                if next_feed[0, 0] == configs.named_constants.KOMYEOS:  # in decoder vocab
                    print("<KOMYEOS>", end=" ")
                elif next_feed[0, 0] == configs.named_constants.KOMYSOS:  # in decoder vocab
                    print("<KOMYSOS>", end=" ")
                elif next_feed[0, 0] == configs.named_constants.KOMYPAD:  # in decoder vocab
                    print("<KOMYPAD>", end=" ")
                else:
                    print(next_feed[0, 0] - 1, end=" ")  # in encoder vocab

                if j < test_sequence.shape[0] and (test_sequence_target[j][0] == sample - 1 or test_sequence_target[j][0] == sample == configs.named_constants.KOMYEOS):
                    cnt += 1

                j += 1

            print("({}/{})".format(cnt, test_sequence.shape[0]), ".")
            print("*" * 50)

    def end_epoch_callback(self, graph: ModelGraph, configs: ModelConfigs, sess: tf.Session):
        pass

    def test_callback(self, graph: ModelGraph, configs: ModelConfigs, sess: tf.Session):
        self.step = 0
        test_data_generator = DataGenerator(configs, "not train")

        # accumulators to average
        valid_logprob = 0
        valid_loss = 0
        ts_accuracy = 0

        for _, t_encoder_in, t_decoder_in, t_decoder_out, t_encoder_decoder_len in test_data_generator.next_batch(configs.batch_size):
            self.step += 1
            feed_dict = {graph.keep_prop_tf: 1.0,
                         graph.encoder_inputs: t_encoder_in,
                         graph.encoder_inputs_length: t_encoder_decoder_len,
                         graph.decoder_inputs: t_decoder_in,
                         graph.decoder_inputs_length: t_encoder_decoder_len,
                         graph.decoder_outputs: t_decoder_out}

            loss_, smm, accuracy_, prediction_probabilities = sess.run([graph.loss, graph.summaries, graph.accuracy, graph.dec_probabilities], feed_dict=feed_dict)

            ts_accuracy += accuracy_
            valid_loss += loss_
            valid_logprob = valid_logprob + negative_log_prob(prediction_probabilities, t_decoder_out)

            if not self.is_local_env:
                self.validation_writer.add_summary(smm, self.summary_global_step)

        # average your values
        v_perplexity = float(2 ** (valid_logprob / self.step))  # our base is 2 we also use log2 at negativeLogProb

        if not self.is_local_env:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="perplexity", simple_value=v_perplexity),
            ])

            self.validation_writer.add_summary(summary, self.summary_global_step)

        valid_loss /= self.step
        ts_accuracy /= self.step

        self.logger.log_validation(ts_accuracy, valid_loss, v_perplexity)

        self.step = 0
