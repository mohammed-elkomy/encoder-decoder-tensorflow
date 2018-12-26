import time

import numpy as np

from model_configs import ModelConfigs

'''
bucketed data generator implementation for seq2seq model as if we have source language with vocabulary_size_encoder and destination language 
the relation between the two languages is just reverse relationship
'''


class DataGenerator:
    def __init__(self, manager, generator="train"):
        np.random.seed(int(time.time()))

        configs: ModelConfigs = manager.configs
        if generator == "train":
            manager.set_data_generator(self)
        self.manager = manager
        self.generator = generator
        #############################################################################

        if generator == "train":
            self.samples = configs.num_samples_train
            self.buckets = configs.num_buckets_train
            self.num_epochs = configs.num_epochs
        else:
            self.samples = configs.num_samples_test
            self.buckets = configs.num_buckets_test
            self.num_epochs = 1

        self.max_len = configs.max_seq_len_encoder
        self.batch_size = configs.batch_size
        self.vocab = configs.vocabulary_size_encoder

        self.special_chars = 2  # <KOMYPAD>=0 <KOMYEOS>=1
        self.current_epoch = 0

        self.named_constans = configs.named_constants

        self.buckets_cursors = np.array([0] * self.buckets)
        self.samples_per_bucket = self.samples // self.buckets
        self.bucketed_data, self.bucketed_length = self.gen_dataset()

    def gen_dataset(self):
        # you need uniform distribution of course :3
        length = np.random.randint(3, self.max_len + 1, self.samples)  # exclusive
        random_data_block = np.random.randint(self.special_chars, self.vocab, (self.samples, self.max_len))  # just don't make 0 since it's pad token 1 for end of sequence

        sorted_indx = np.argsort(length)

        random_data_block = random_data_block[sorted_indx]
        length = length[sorted_indx]

        bucketed_data = []
        bucketed_length = []

        for i in range(self.buckets):
            bucket_permutation = np.random.permutation(range(i * self.samples_per_bucket, (i + 1) * self.samples_per_bucket))
            bucketed_data.append(random_data_block[bucket_permutation])
            bucketed_length.append(length[bucket_permutation])

        return bucketed_data, bucketed_length

    def reset_data(self):
        self.bucketed_data, self.bucketed_length = self.gen_dataset()
        self.buckets_cursors = np.array([0] * self.buckets)

    def next_batch(self):
        while self.num_epochs > self.current_epoch:  # generates batch at each iteration and checking the epoch
            is_end_of_epoch = False

            chosen_bucket = np.random.randint(0, self.buckets)
            while self.buckets_cursors[chosen_bucket] + self.batch_size > self.samples_per_bucket:
                chosen_bucket = np.random.randint(0, self.buckets)

            batch_data = self.bucketed_data[chosen_bucket][self.buckets_cursors[chosen_bucket]:self.buckets_cursors[chosen_bucket] + self.batch_size]  # sublist of bucket list
            batch_lengths = self.bucketed_length[chosen_bucket][self.buckets_cursors[chosen_bucket]:self.buckets_cursors[chosen_bucket] + self.batch_size]
            self.buckets_cursors[chosen_bucket] += self.batch_size

            # Pad sequences with 0s so they are all the same length
            maxlen = max(batch_lengths)
            feed_batch = np.zeros([self.batch_size, maxlen], dtype=np.int32)  # zeros to leave the padding after flushing the data
            rev_feed_batch = np.zeros([self.batch_size, maxlen + 1], dtype=np.int32)  # add 1 for the start of sequence

            for i, x_i in enumerate(feed_batch):
                x_i[:batch_lengths[i] - 1] = batch_data[i][:batch_lengths[i] - 1]  # process each sample in batch
                x_i[batch_lengths[i] - 1] = self.named_constans.KOMYEOS

            batch_data += 1  # make sure there is no special_tokens in the target text since i need to PAD,SOS,EOS

            # just put the start maker and place the source data but flipped
            rev_feed_batch[:, 0] = self.named_constans.KOMYSOS  # start token is one
            for i, y_i in enumerate(rev_feed_batch):
                # add one and sub one :3 this means i need len -1 elements as the last one is the EOS .. starting from position 1 because of the SOS

                # # here we flip
                y_i[1:batch_lengths[i] + 1 - 1] = np.flip(batch_data[i][:batch_lengths[i] - 1], axis=0)  # process each sample in batch
                # # no flip
                # y_i[1:batch_lengths[i] + 1 - 1] = batch_data[i][:batch_lengths[i] - 1]  # process each sample in batch

                y_i[batch_lengths[i] + 1 - 1] = self.named_constans.KOMYEOS

            if np.all(self.buckets_cursors + self.batch_size > self.samples_per_bucket):  # buckets_cursors are 0 based
                # for any cursor if the bucket step size is exceeded just reset, this will tend to be normal with large numbers
                is_end_of_epoch = True
                self.reset_data()
                if self.generator != "train":
                    break

            # is_end_of_epoch,encoder_in, decoder_in,decoder_out,encoder_len,decoder_len
            yield is_end_of_epoch, feed_batch.transpose(), rev_feed_batch[:, :-1].transpose(), rev_feed_batch[:, 1:].transpose(), batch_lengths  # time major = seqlen * batch

    def average_padding(self):
        # average padding in the dataset after bucketing
        padding = 0
        num_batches = 0

        for _, _, _, _, lengths in self.next_batch():
            if num_batches > 5000:
                break
            num_batches += 1
            max_len = max(lengths)
            padding += np.sum(max_len - lengths)
        print("Average padding per batch:", padding / num_batches, "as the batch has", self.batch_size * self.max_len, "tokens", "padding percentage:", (padding / num_batches) / (self.batch_size * self.max_len))

# considerations
# encoder_inputs [max_encoder_time, batch_size]: source input words.
# decoder_inputs [max_decoder_time, batch_size]: target input words.
# decoder_outputs [max_decoder_time, batch_size]: target output words, these are decoder_inputs shifted to the left by one time step with an end-of-sentence tag appended on the right.

# in testing we infer by sampling word by word

# attention
# for long sentences, the single fixed-size hidden state becomes an information bottleneck. Instead of discarding all of the hidden states computed in the source RNN, the attention mechanism provides an approach that allows the decoder to peek at them (treating them as a dynamic memory of the source information).

# usage example
# dataGenerator = DataGenerator(num_epochs=num_epochs, vocab=vocabulary_size_encoder, max_len=max_seq_len_encoder, samples=num_samples_train, buckets=num_buckets_train, batch_size=batch_size, special_chars=2)
# dataGenerator.average_padding()
