import numpy as np

from configs import vocabulary_size_encoder, max_seq_len_encoder, num_samples_train, num_buckets_train, batch_size, num_epochs, KOMYSOS, KOMYEOS

'''
bucketed data generator implementation for seq2seq model as if we have source language with vocabulary_size_encoder and destination language 
the relation between the two languages is just reverse relationship
'''


class DataGenerator:
    def __init__(self, num_epochs, vocab, samples, max_len, buckets, batch_size, special_chars):
        self.samples = samples
        self.max_len = max_len
        self.num_epochs = num_epochs
        self.special_chars = special_chars
        self.batch_size = batch_size
        self.current_epoch = 0
        self.vocab = vocab
        self.buckets = buckets
        self.buckets_cursors = np.array([0] * buckets)
        self.samples_per_bucket = self.samples // self.buckets
        self.bucketed_data, self.bucketed_length = self.gen_dataset()

    def gen_dataset(self):
        length = np.random.randint(self.max_len // 2, self.max_len + 1, self.samples)  # exclusive
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

    def next_batch(self, batch):
        while self.num_epochs > self.current_epoch:  # generates batch at each iteration and checking the epoch
            is_end_of_epoch = False
            if np.any(self.buckets_cursors + batch + 1 > self.samples_per_bucket):  # buckets_cursors are 0 based
                # for any cursor if the bucket step size is exceeded just reset, this will tend to be normal with large numbers
                self.current_epoch += 1
                self.reset_data()
                is_end_of_epoch = True

            chosen_bucket = np.random.randint(0, self.buckets)

            batch_data = self.bucketed_data[chosen_bucket][self.buckets_cursors[chosen_bucket]:self.buckets_cursors[chosen_bucket] + batch]
            batch_lengths = self.bucketed_length[chosen_bucket][self.buckets_cursors[chosen_bucket]:self.buckets_cursors[chosen_bucket] + batch]
            self.buckets_cursors[chosen_bucket] += batch

            # Pad sequences with 0s so they are all the same length
            maxlen = max(batch_lengths)
            feed_batch = np.zeros([batch, maxlen], dtype=np.int32)  # zeros to leave the padding after flushing the data
            rev_feed_batch = np.zeros([batch, maxlen + 1], dtype=np.int32)  # add 1 for the start of sequence

            for i, x_i in enumerate(feed_batch):
                x_i[:batch_lengths[i] - 1] = batch_data[i][:batch_lengths[i] - 1]  # process each sample in batch
                x_i[batch_lengths[i] - 1] = KOMYEOS
                # x_i[:batch_lengths[i]] = batch_data[i][:batch_lengths[i]]  # process each sample in batch

            batch_data += 1  # make sure there is no special_tokens in the target text since i need to pad,SOS,EOS

            # just put the start maker and place the source data but flipped
            rev_feed_batch[:, 0] = KOMYSOS  # start token is one
            for i, y_i in enumerate(rev_feed_batch):
                # add one and sub one :3 this means i need len -1 elements as the last one is the EOS .. starting from position 1 because of the SOS
                y_i[1:batch_lengths[i] + 1 - 1] = np.flip(batch_data[i][:batch_lengths[i] - 1], axis=0)  # process each sample in batch
                y_i[batch_lengths[i] + 1 - 1] = KOMYEOS
                # y_i[1:batch_lengths[i] + 1] = np.flip(batch_data[i][:batch_lengths[i]], axis=0)  # process each sample in batch

            # is_end_of_epoch,encoder_in, decoder_in,decoder_out,encoder_len,decoder_len
            yield is_end_of_epoch, feed_batch.transpose(), rev_feed_batch[:, :-1].transpose(), rev_feed_batch[:, 1:].transpose(), batch_lengths, batch_lengths  # time major = seqlen * batch

    def average_padding(self):
        # average padding in the dataset after bucketing
        padding = 0
        num_batches = 1000
        for i in range(num_batches):
            lengths = next(self.next_batch(self.batch_size))[4]
            max_len = max(lengths)
            padding += np.sum(max_len - lengths)
        print("Average padding / batch:", padding / num_batches, "as the batch has", self.batch_size * self.max_len, "tokens", "padding percentage:", (padding / num_batches) / (self.batch_size * self.max_len))


# considerations
# encoder_inputs [max_encoder_time, batch_size]: source input words.
# decoder_inputs [max_decoder_time, batch_size]: target input words.
# decoder_outputs [max_decoder_time, batch_size]: target output words, these are decoder_inputs shifted to the left by one time step with an end-of-sentence tag appended on the right.

# in testing we infer by sampling word by word

# attention
# for long sentences, the single fixed-size hidden state becomes an information bottleneck. Instead of discarding all of the hidden states computed in the source RNN, the attention mechanism provides an approach that allows the decoder to peek at them (treating them as a dynamic memory of the source information).


# usage example
dataGenerator = DataGenerator(num_epochs=num_epochs, vocab=vocabulary_size_encoder, max_len=max_seq_len_encoder, samples=num_samples_train, buckets=num_buckets_train, batch_size=batch_size, special_chars=2)
dataGenerator.average_padding()
