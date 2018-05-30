import sys

from Logger_Helper import print_hyper

# Model hyper-parameters
KOMYSOS = 2
KOMYEOS = 1
KOMYPAD = 0
vocabulary_size_encoder = 99  # this includes one special tokens <KOMYPAD>=0 <KOMYEOS>=1 .........0 >TO> 98
vocabulary_size_decoder = 100  # this includes 2 special tokens <KOMYPAD>=0 <KOMYEOS>=1 <KOMYSOS>=2  ..........0 >TO> 99

max_seq_len_encoder = 96
max_seq_len_decoder = 108

num_samples_train = 1024 * 256 * 16
num_samples_test = 1024 * 16 * 20
num_buckets_train = 16
num_buckets_test = 4
batch_size = 256

assert (num_samples_train // num_buckets_train) % batch_size == 0
assert (num_samples_test // num_buckets_test) % batch_size == 0

use_embedding = True
encoder_embedding_size = 64
decoder_embedding_size = 64

stacked_layers = 2
keep_prop = .2
internal_state_encoder = 512
internal_state_decoder = internal_state_encoder * 2

num_epochs = 10

learning_rate = 1e-3
##############################################
# training system configs
minute = 60
summary_every_mins = 3
chechpoint_every_mins = 30

is_local_env = True
if is_local_env:
    num_samples_train = 256 * 16 * 8
    num_samples_test = 256 * 4 * 8
    minute = 0
##############################################
# system summary

outputFileName = 'nmt.log'

if not is_local_env:
    sys.stdout = open(outputFileName, 'w')


def print_init_configs():
    print("{} data: {} samples,each bucket has {} samples , each bucket has {} batches"
          .format("train", num_samples_train, num_samples_train // num_buckets_train, num_samples_train // num_buckets_train // batch_size))
    print("{} data: {} samples,each bucket has {} samples , each bucket has {} batches"
          .format("test", num_samples_test, num_samples_test // num_buckets_test, num_samples_test // num_buckets_test // batch_size))

    print_hyper({
        "vocabulary_size_encoder": vocabulary_size_encoder,
        "vocabulary_size_decoder": vocabulary_size_decoder,
        "max_seq_len_encoder": max_seq_len_encoder,
        "max_seq_len_decoder": max_seq_len_decoder,
        "num_samples_train": num_samples_train,
        "num_samples_test": num_samples_test,
        "num_buckets_train": num_buckets_train,
        "num_buckets_test": num_buckets_test,
        "batch_size": batch_size,
        "encoder_embedding_size": encoder_embedding_size,
        "decoder_embedding_size": decoder_embedding_size,
        "stacked_layers": stacked_layers,
        "keep_prop": keep_prop,
        "internal_state_encoder": internal_state_encoder,
        "internal_state_decoder": internal_state_decoder,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
    })


sys.stdout.flush()
