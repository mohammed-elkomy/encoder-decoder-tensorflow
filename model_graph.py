import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from base_ModelGraph import BaseModelGraph
from base_TrainingManager import BaseTrainingManager


class ModelGraph(BaseModelGraph):

    def __init__(self, base_training_manager: BaseTrainingManager):
        super(ModelGraph, self).__init__(base_training_manager)
        base_training_manager.set_graph(self)

    def make_saver(self):
        self.saver = tf.train.Saver(max_to_keep=1)  # reminder : saver is an op

    def variable_initializer(self):
        self.init = tf.global_variables_initializer()

    def build_graph(self):
        configs = self.trainingManager.configs

        # shared between train and test
        self.keep_prop_tf = tf.placeholder(dtype=tf.float32, name="keep_prop_tf")

        # repeat it stacked_layers
        encoder_dropcells_fw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(configs.internal_state_encoder), output_keep_prob=self.keep_prop_tf) for _ in range(configs.stacked_layers)]
        encoder_multi_cell_fw = rnn.MultiRNNCell(encoder_dropcells_fw, state_is_tuple=True)

        encoder_dropcells_bw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(configs.internal_state_encoder), output_keep_prob=self.keep_prop_tf) for _ in range(configs.stacked_layers)]
        encoder_multi_cell_bw = rnn.MultiRNNCell(encoder_dropcells_bw, state_is_tuple=True)

        decoder_dropcells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(configs.internal_state_decoder), output_keep_prob=self.keep_prop_tf) for _ in range(configs.stacked_layers)]
        decoder_multi_cell = rnn.MultiRNNCell(decoder_dropcells, state_is_tuple=True)

        with tf.variable_scope('train'):
            # input placeholders
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')  # [en_seq_len, batch  ]
            self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')  # [batch]

            self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')  # [de_seq_len, batch] starts with KOKO_START token
            self.decoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_inputs_length')  # [batch]  IMPORTANT NOTE : decoder_inputs_length = counts the start token
            self.decoder_outputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_outputs')  # [de_seq_len, batch ] = ground truth padded at the end with zeros

            # embedding lookup or one hot encoding of my languages (encoder / decoder)
            if configs.use_embedding:
                encoder_embeddings = tf.Variable(tf.random_uniform([configs.vocabulary_size_encoder, configs.encoder_embedding_size], -1.0, 1.0), dtype=tf.float32)
                encoder_inputs_to_rnn = tf.nn.embedding_lookup(encoder_embeddings, self.encoder_inputs)  # [  sequence_length,batch_size, encoder_embedding_size ] # embedded

                decoder_embeddings = tf.Variable(tf.random_uniform([configs.vocabulary_size_decoder, configs.decoder_embedding_size], -1.0, 1.0), dtype=tf.float32)
                decoder_inputs_to_rnn = tf.nn.embedding_lookup(decoder_embeddings, self.decoder_inputs)  # [  sequence_length,batch_size, decoder_embedding_size ] # embedded

            else:
                encoder_inputs_to_rnn = tf.one_hot(self.encoder_inputs, configs.vocabulary_size_encoder, 1.0, 0.0)  # [  sequence_length,batch_size, vocabulary_size ]  # one hot encoded
                decoder_inputs_to_rnn = tf.one_hot(self.decoder_inputs, configs.vocabulary_size_decoder, 1.0, 0.0)  # [  sequence_length,batch_size, vocabulary_size ]  # one hot encoded

            (self.encoder_fw_outputs, self.encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_multi_cell_fw,
                                                cell_bw=encoder_multi_cell_bw,
                                                inputs=encoder_inputs_to_rnn,
                                                sequence_length=self.encoder_inputs_length,
                                                dtype=tf.float32,
                                                time_major=True)
            # outputs :[sequence_length, batch_size, internal_state_encoder]
            # final_state:[batch_size, internal_state_encoder] as tuple repeated stack times
            # this is my thought vector = [batch_size, internal_state_encoder(fw)+internal_state_encoder(bw)]

            # i will re-feed this thought vector at inference
            self.decoder_current_state = tuple([tf.concat((encoder_fw_final_state[i], encoder_bw_final_state[i]), axis=1) for i in range(configs.stacked_layers)])  # state is tuple for decoder input

            # decoder dynamic rnn
            self.decoder_states_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(decoder_multi_cell,
                                                                                      inputs=decoder_inputs_to_rnn,
                                                                                      initial_state=self.decoder_current_state,
                                                                                      time_major=True,
                                                                                      sequence_length=self.decoder_inputs_length)
            # decoder_states_outputs :[sequence_length, batch_size, internal_state_decoder]
            # decoder_final_state :[batch_size, internal_state_decoder] as tuple repeated stack times

            decoder_logits = tf.layers.dense(self.decoder_states_outputs, units=configs.vocabulary_size_decoder, use_bias=True)  # projection on the vocabulary outputs : [sequence_length, batch_size, vocabulary_size_decoder]
            self.dec_probabilities = tf.nn.softmax(decoder_logits)  # [sequence_length, batch_size, vocabulary_size_decoder]

            # the triangle has the decoder shape not the encoder !!!!
            lower_triangular_ones = tf.constant(np.tril(np.ones([configs.max_seq_len_decoder, configs.max_seq_len_decoder])), dtype=tf.float32)  # lower triangle ones [max_seq_len_encoder,max_seq_len_encoder] >> [[1. 0.],[1. 1.]]
            _, batch_size_tf = tf.unstack(tf.shape(self.encoder_inputs))  # seq_length , batch_size

            seqlen_mask = tf.transpose(tf.slice(tf.gather(lower_triangular_ones, self.decoder_inputs_length - 1), begin=[0, 0], size=[batch_size_tf, tf.reduce_max(self.decoder_inputs_length)]))  # so you need to take length -1 due to lower triangle ones [sequence_length, batch_size]

            # connect outputs to
            with tf.name_scope("optimization"):
                # Loss function
                self.loss = tf.contrib.seq2seq.sequence_loss(decoder_logits, self.decoder_outputs, seqlen_mask)  # sparse softmax cross entropy

                # Optimizer
                self.train_step = tf.train.RMSPropOptimizer(configs.learning_rate).minimize(self.loss)

            # To calculate the number correct, this means we don't count the padded as correct
            correct = tf.cast(tf.equal(tf.cast(tf.argmax(decoder_logits, 2), tf.int32), self.decoder_outputs), dtype=tf.float32) * seqlen_mask
            self.accuracy = tf.reduce_sum(correct) / tf.reduce_sum(seqlen_mask)

            # summary tensors
            if not self.trainingManager.is_local_env:
                loss_summary = tf.summary.scalar("batch_loss", self.loss)
                acc_summary = tf.summary.scalar("batch_accuracy", self.accuracy)
                self.summaries = tf.summary.merge([loss_summary, acc_summary])
