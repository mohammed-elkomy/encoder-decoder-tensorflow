import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn import raw_rnn

from ..base_ModelGraph import BaseModelGraph
from ..base_TrainingManager import BaseTrainingManager


class ModelGraphPureAtt(BaseModelGraph):

    def __init__(self, base_training_manager: BaseTrainingManager):
        super(ModelGraphPureAtt, self).__init__(base_training_manager)

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

            (self.memory_fw, self.memory_bw), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_multi_cell_fw,
                                                cell_bw=encoder_multi_cell_bw,
                                                inputs=encoder_inputs_to_rnn,
                                                sequence_length=self.encoder_inputs_length,
                                                dtype=tf.float32,
                                                time_major=True, swap_memory=True)
            # outputs :[sequence_length, batch_size, internal_state_encoder*stackedlayers]
            # final_state:[batch_size, internal_state_encoder] as tuple repeated stack times
            # this is my thought vector = [batch_size, internal_state_encoder(fw)+internal_state_encoder(bw)]

            # memory for attention
            # self.memory_ = tf.concat((self.memory_fw, self.memory_bw), axis=2)  # outputs :[sequence_length, batch_size, internal_state_encoder*2]
            # i will split them

            # i will re-feed this thought vector at inference
            self.decoder_init_state_from_encoder = tuple([tf.concat((encoder_fw_final_state[i], encoder_bw_final_state[i]), axis=1) for i in range(configs.stacked_layers)])  # state is tuple for decoder input

            # # decoder dynamic rnn
            # self.decoder_states_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(decoder_multi_cell,
            #                                                                             inputs=decoder_inputs_to_rnn,
            #                                                                             initial_state=self.decoder_current_state,
            #                                                                             time_major=True,
            #                                                                             sequence_length=self.decoder_inputs_length)

            # # my decoder dynamic rnn1
            # self.decoder_states_outputs, self.decoder_final_state = self.my_dynamic_rnn(decoder_multi_cell,
            #                                                                             inputs=decoder_inputs_to_rnn,
            #                                                                             initial_state=self.decoder_current_state,
            #                                                                             sequence_length=self.decoder_inputs_length)

            # # my decoder dynamic rnn2
            # self.decoder_states_outputs, self.decoder_final_state, self.loop_outputs = self.my_dynamic_rnn_stacked_out(decoder_multi_cell,
            #                                                                                                            inputs=decoder_inputs_to_rnn,
            #                                                                                                            initial_state=self.decoder_current_state,
            #                                                                                                            sequence_length=self.decoder_inputs_length)

            # # my decoder dynamic rnn2
            self.decoder_states_outputs, self.decoder_final_state, self.loop_outputs = self.my_attentive(decoder_multi_cell,
                                                                                                         inputs=decoder_inputs_to_rnn,
                                                                                                         encoder_final_state=self.decoder_init_state_from_encoder,
                                                                                                         sequence_length=self.decoder_inputs_length,
                                                                                                         memory_bw=self.memory_bw, memory_fw=self.memory_fw)

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
                # regularize = tf.contrib.layers.l2_regularizer(.5)
                # params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                # reg_term = sum([regularize(param) for param in params])
                # self.loss += reg_term

                # Optimizer
                optimizer = tf.train.RMSPropOptimizer(configs.learning_rate)

                # returns grads_and_vars is a list of tuples [(gradient, variable)]
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                # check this
                # zip([(2,3),(4,5),(4,5),(4,5),(4,5)])  will be <zip at 0xb3de450648> and as list [((2, 3),), ((4, 5),), ((4, 5),), ((4, 5),), ((4, 5),)] !!

                # a,b=zip(*[(2,3),(4,5),(4,5),(4,5),(4,5)]) will be ((2, 4, 4, 4, 4),(3, 5, 5, 5, 5))
                # the same as so * is unpack operator
                # a,b=zip((2,3),(4,5),(4,5),(4,5),(4,5))

                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1)  # 1 to 5
                # computes the global norm and then shrink all gradients with the same ratio clip_norm/global_norm only if global_norm > clip_norm

                self.train_step = optimizer.apply_gradients(list(zip(gradients, variables)))  # zip to relate variable to gradient as list of tuples

            # To calculate the number correct, this means we don't count the padded as correct
            correct = tf.cast(tf.equal(tf.cast(tf.argmax(decoder_logits, 2), tf.int32), self.decoder_outputs), dtype=tf.float32) * seqlen_mask
            self.accuracy = tf.reduce_sum(correct) / tf.reduce_sum(seqlen_mask)

            # summary tensors
            if not self.trainingManager.is_local_env:
                loss_summary = tf.summary.scalar("batch_loss", self.loss)
                acc_summary = tf.summary.scalar("batch_accuracy", self.accuracy)
                self.summaries = tf.summary.merge([loss_summary, acc_summary])

    # assumed to be time major
    def my_dynamic_rnn(self, cell, sequence_length, inputs, initial_state):  # initial_state = final state of encoder
        inputs_shape = tf.shape(inputs)
        max_seq_len, batch_size, input_features = self.trainingManager.configs.max_seq_len_decoder, inputs_shape[1], inputs.shape[2]

        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_seq_len)  # max_length = time
        inputs_ta = inputs_ta.unstack(inputs)  # length array of [batch , hidden state]

        def loop_fn(cur_time, cur_cell_output, cur_cell_state, cur_loop_state):  # current inputs
            nxt_emit_output = cur_cell_output  # == None for time == 0

            if cur_cell_output is None:  # time == 0
                # initialization logic
                nxt_cell_state = initial_state
            else:
                # any logic that depends on the cell state or cell output..ex attention
                # this part is 1 based
                nxt_cell_state = cur_cell_state

            # common loop logic
            # as in traditional loop the condition is "cur_time < sequence_length" but here i want the finished
            cur_elements_finished = (cur_time >= sequence_length)  # [batch] # this part is 0 based

            is_current_out_of_bound = tf.reduce_all(cur_elements_finished)  # scalar --  will cut to the longest sequence given for example [5,2,f] with lengths [3,4] will end at 4

            # this shape has to be deterministic not [....,?]
            nxt_input = tf.cond(
                is_current_out_of_bound,
                lambda: tf.zeros([batch_size, input_features],  # input shape [batch , input_features]
                                 dtype=tf.float32),  # no input for end of loop .. can't read if out of bounds == time

                lambda: inputs_ta.read(cur_time)  # read current input
            )

            nxt_loop_state = None
            return cur_elements_finished, nxt_input, nxt_cell_state, nxt_emit_output, nxt_loop_state  # next step in time

        outputs_ta, final_state, _ = raw_rnn(cell, loop_fn)
        outputs = outputs_ta.stack()  # [seq_len, batch, hidden_state]
        # final_state # ([batch, hidden_state]) stacked times
        return outputs, final_state

    # assumed to be time major
    # returns stacked hidden state
    def my_dynamic_rnn_stacked_out(self, cell, sequence_length, inputs, initial_state):  # initial_state = final state of encoder
        inputs_shape = tf.shape(inputs)
        max_seq_len, batch_size, input_features = self.trainingManager.configs.max_seq_len_decoder, inputs_shape[1], inputs.shape[2]

        stacked_layers = self.trainingManager.configs.stacked_layers

        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_seq_len)  # max_length = time
        inputs_ta = inputs_ta.unstack(inputs)  # length array of [batch , hidden state]

        output_stacked_states_ta = tf.TensorArray(size=max_seq_len, dtype=tf.float32)  # max_length = tim

        zero_state_tuple = cell.zero_state(batch_size, dtype=tf.float32)  # cell zero state

        def loop_fn(cur_time, cur_cell_output, cur_cell_state, cur_loop_state):  # current inputs
            nxt_emit_output = cur_cell_output  # == None for time == 0

            if cur_cell_output is None:  # time == 0
                # initialization logic
                nxt_cell_state = initial_state
                nxt_loop_state = output_stacked_states_ta
            else:
                # any logic that depends on the cell state or cell output..ex attention
                # this part is 1 based
                nxt_cell_state = cur_cell_state
                nxt_loop_state = cur_loop_state.write(cur_time - 1,
                                                      tuple([
                                                          tf.where(cur_time - 1 < sequence_length,  # this part is 1 based
                                                                   cur_cell_state[i],
                                                                   zero_state_tuple[i])
                                                          for i in range(stacked_layers)
                                                      ]))

            # common loop logic
            # as in traditional loop the condition is "cur_time < sequence_length" but here i want the finished
            cur_elements_finished = (cur_time >= sequence_length)  # [batch] # this part is 0 based

            is_current_out_of_bound = tf.reduce_all(cur_elements_finished)  # scalar --  will cut to the longest sequence given for example [5,2,f] with lengths [3,4] will end at 4

            # this shape has to be deterministic not [....,?]
            nxt_input = tf.cond(
                is_current_out_of_bound,
                lambda: tf.zeros([batch_size, input_features],  # input shape [batch , input_features]
                                 dtype=tf.float32),  # no input for end of loop .. can't read if out of bounds == time

                lambda: inputs_ta.read(cur_time)  # read current input
            )

            # nxt_loop_state = None
            return cur_elements_finished, nxt_input, nxt_cell_state, nxt_emit_output, nxt_loop_state  # next step in time

        outputs_ta, final_state, loop_ta = raw_rnn(cell, loop_fn)
        outputs = outputs_ta.stack()  # [seq_len, batch, hidden_state ]
        loop = loop_ta.stack()  # [seq_len, stacked_layers, batch, hidden_state]
        loop = tf.transpose(loop, [0, 2, 1, 3])  # [seq_len,batch,stacked_layers,hidden_state ]
        loop = tf.reshape(loop, [-1, batch_size, stacked_layers * cell.output_size])  # [seq_len, batch, stacked_layers x hidden_state ]
        return outputs, final_state, loop

        # assumed to be time major
        # returns attention weights too
        # i will transform my context vector to embedding size to exactly 64 since my pure attention vector is 2048

    def my_attentive_concat_memory(self, cell, sequence_length, inputs, encoder_final_state, memory):
        inputs_shape = tf.shape(inputs)
        max_seq_len, batch_size, input_features = self.trainingManager.configs.max_seq_len_decoder, inputs_shape[1], inputs.shape[2]

        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_seq_len)  # max_length = time
        inputs_ta = inputs_ta.unstack(inputs)  # length array of [batch , hidden state]

        attention_weights = tf.TensorArray(size=max_seq_len, dtype=tf.float32)  # max_length = tim

        def loop_fn(cur_time, cur_cell_output, cur_cell_state, cur_loop_state):  # current inputs
            nxt_emit_output = tf.zeros([cell.output_size * 2],
                                       dtype=tf.float32)  # define initial size of output or the default is cell , dont give batch size !!!!!!

            # nxt_emit_output = None
            compressed_context_vector = tf.zeros([batch_size, input_features],  # [batch , input_features]
                                                 dtype=tf.float32)

            if cur_cell_output is None:  # time == 0
                # initialization logic
                nxt_cell_state = encoder_final_state
                nxt_loop_state = attention_weights
            else:
                # any logic that depends on the cell state or cell output..ex attention
                # this part is 1 based
                nxt_cell_state = cur_cell_state  # [batch , stacked*hidden_decoder=(1024*2)]

                scalars = tf.reduce_sum(
                    tf.multiply(memory, cur_cell_state[-1]),  # [seq_len_encoder, batch ,stacked*hidden_encoder=(512*2)] , mul by top state
                    axis=2)  # [seq_len_encoder, batch ] this is cross product

                scalars = tf.transpose(tf.nn.softmax(scalars, axis=0))  # [batch,seq_len_encoder] this is cross product

                memory_trans = tf.transpose(memory, [2, 1, 0])  # [stacked*hidden_encoder=(512*2), batch,seq_len_encoder]

                pure_context_vector = tf.reduce_sum(
                    tf.transpose(
                        tf.multiply(memory_trans, scalars),  # [stacked*hidden_encoder=(512*2),batch,seq_len_encoder ]
                        [2, 1, 0]),  # [seq_len_encoder, batch ,stacked*hidden_encoder=(512*2)]
                    axis=0)  # [batch ,stacked*hidden_encoder=(512*2)]

                compressed_context_vector = tf.layers.dense(pure_context_vector, units=input_features)  # [batch ,seq_len_encoder]

                nxt_emit_output = tf.concat((cur_cell_output, pure_context_vector), axis=1)  # [batch ,2*stacked*hidden_encoder=(512*2*2)]
                nxt_loop_state = cur_loop_state.write(cur_time - 1,
                                                      tf.where(cur_time - 1 < sequence_length,  # this part is 1 based
                                                               scalars,
                                                               tf.zeros_like(scalars)))

            # common loop logic
            # as in traditional loop the condition is "cur_time < sequence_length" but here i want the finished
            cur_elements_finished = (cur_time >= sequence_length)  # [batch] # this part is 0 based

            is_current_out_of_bound = tf.reduce_all(cur_elements_finished)  # scalar --  will cut to the longest sequence given for example [5,2,f] with lengths [3,4] will end at 4

            # this shape has to be deterministic not [....,?]
            nxt_input = tf.cond(
                is_current_out_of_bound,
                lambda: tf.zeros([batch_size, input_features * 2],  # input shape [batch , input_features+input_features]
                                 dtype=tf.float32),  # no input for end of loop .. can't read if out of bounds == time

                lambda: tf.concat((inputs_ta.read(cur_time), compressed_context_vector), axis=1)  # read current input and concat context vector
            )

            # nxt_loop_state = None
            return cur_elements_finished, nxt_input, nxt_cell_state, nxt_emit_output, nxt_loop_state  # next step in time

        outputs_ta, final_state, loop_ta = raw_rnn(cell, loop_fn, swap_memory=True)
        outputs = outputs_ta.stack()  # [seq_len_decoder_decoder, batch, hidden_state ]
        loop = loop_ta.stack()  # [seq_len_decoder_decoder, seq_len_encoder(memory), batch]
        return outputs, final_state, loop

    # assumed to be time major
    # returns attention weights too
    # i will transform my context vector to embedding size to exactly 64 since my pure attention vector is 2048
    def my_attentive(self, cell, sequence_length, inputs, encoder_final_state, memory_fw, memory_bw):
        inputs_shape = tf.shape(inputs)
        max_seq_len, batch_size, input_features = self.trainingManager.configs.max_seq_len_decoder, inputs_shape[1], inputs.shape[2]

        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_seq_len)  # max_length = time
        inputs_ta = inputs_ta.unstack(inputs)  # length array of [batch , hidden state]

        attention_weights = tf.TensorArray(size=max_seq_len, dtype=tf.float32)  # max_length = time

        def loop_fn(cur_time, cur_cell_output, cur_cell_state, cur_loop_state):  # current inputs
            # cur_cell_output = None at time = 0
            nxt_emit_output = tf.zeros([cell.output_size * 2],  # (decoder hidden = 512 )+(context vector_fw = 256)+(context vector_bw = 256)
                                       dtype=tf.float32)  # define initial size of output or the default is cell , dont give batch size !!!!!!

            compressed_context_vector_fw = tf.zeros([batch_size, input_features // 2],  # [batch , input_features/2]
                                                    dtype=tf.float32)

            compressed_context_vector_bw = tf.zeros([batch_size, input_features // 2],  # [batch , input_features/2]
                                                    dtype=tf.float32)

            if cur_cell_output is None:  # time == 0
                # initialization logic
                nxt_cell_state = encoder_final_state
                nxt_loop_state = attention_weights
            else:
                # any logic that depends on the cell state or cell output..ex attention
                # this part is 1 based
                nxt_cell_state = cur_cell_state  # [batch , stacked*hidden_decoder=(512*2)]

                pure_context_vector_fw, scalars_fw = self.attention_step(cur_cell_state, memory_fw)  # [batch ,hidden_encoder]
                pure_context_vector_bw, scalars_bw = self.attention_step(cur_cell_state, memory_bw)  # [batch ,hidden_encoder]

                compressed_context_vector_fw = tf.layers.dense(pure_context_vector_fw, units=input_features // 2)  # [batch ,input_features/2]
                compressed_context_vector_bw = tf.layers.dense(pure_context_vector_bw, units=input_features // 2)  # [batch ,input_features/2]

                nxt_emit_output = tf.concat((cur_cell_output, pure_context_vector_fw, pure_context_vector_bw), axis=1)  # [batch ,hidden_decoder+hidden_encoder+hidden_encoder=(512+256+256)]

                not_finished = (cur_time - 1 < sequence_length)  # this part is 1 based
                nxt_loop_state = cur_loop_state.write(cur_time - 1,
                                                      (
                                                          tf.where(not_finished,
                                                                   scalars_fw,
                                                                   tf.zeros_like(scalars_fw)
                                                                   ),
                                                          tf.where(not_finished,
                                                                   scalars_bw,
                                                                   tf.zeros_like(scalars_bw)
                                                                   )
                                                      )  # a pair of forward and backward attention weights
                                                      )

            # common loop logic
            # as in traditional loop the condition is "cur_time < sequence_length" but here i want the finished
            cur_elements_finished = (cur_time >= sequence_length)  # [batch] # this part is 0 based

            is_current_out_of_bound = tf.reduce_all(cur_elements_finished)  # scalar --  will cut to the longest sequence given for example [5,2,f] with lengths [3,4] will end at 4

            # this shape has to be deterministic not [....,?]
            nxt_input = tf.cond(
                is_current_out_of_bound,
                lambda: tf.zeros([batch_size, input_features * 2],  # input shape [batch , input_features+input_features]
                                 dtype=tf.float32),  # no input for end of loop .. can't read if out of bounds == time

                lambda: tf.concat((inputs_ta.read(cur_time), compressed_context_vector_fw, compressed_context_vector_bw), axis=1)  # read current input and concat context vector
            )

            # nxt_loop_state = None
            return cur_elements_finished, nxt_input, nxt_cell_state, nxt_emit_output, nxt_loop_state  # next step in time

        outputs_ta, final_state, loop_ta = raw_rnn(cell, loop_fn, swap_memory=True)
        outputs = outputs_ta.stack()  # [seq_len_decoder_decoder, batch, hidden_state+context vector(512+256+256) ]
        loop = loop_ta.stack()  # [seq_len_decoder_decoder, 2, batch,seq_len_encoder(memory)]
        return outputs, final_state, loop

    def attention_step(self, source_hidden_state, memory):
        # memory : [seq_len_encoder, batch , hidden_encoder] , mul by top state
        # source_hidden_state : [batch , hidden_decoder] X stacked
        scalars = tf.reduce_sum(
            tf.multiply(memory, tf.layers.dense(source_hidden_state[-1], units=self.trainingManager.configs.internal_state_encoder)),  # [seq_len_encoder, batch , hidden_encoder] , mul by top state
            axis=2)  # [seq_len_encoder, batch ] this is cross product

        scalars = tf.transpose(tf.nn.softmax(scalars, axis=0))  # [batch,seq_len_encoder]
        memory_trans = tf.transpose(memory, [2, 1, 0])  # [hidden_encoder, batch,seq_len_encoder]

        pure_context_vector = tf.reduce_sum(
            tf.transpose(
                tf.multiply(memory_trans, scalars),  # [hidden_encoder,batch,seq_len_encoder ]
                [2, 1, 0]),  # [seq_len_encoder, batch ,hidden_encoder]
            axis=0)  # [batch ,hidden_encoder]

        return pure_context_vector, scalars
