# =============================Model named constants=============================
class ModelNamedConstant:
    def __init__(self):
        self.KOMYSOS = 2
        self.KOMYEOS = 1
        self.KOMYPAD = 0


class ModelConfigs:
    """any configs related to deep learning(model building + training) is placed here"""

    def __init__(self, vocabulary_size_encoder,
                 vocabulary_size_decoder,
                 max_seq_len_encoder,
                 max_seq_len_decoder,
                 num_samples_train,
                 num_samples_test,
                 num_buckets_train,
                 num_buckets_test,
                 batch_size,
                 use_embedding,
                 encoder_embedding_size,
                 decoder_embedding_size,
                 stacked_layers,
                 keep_prop,
                 internal_state_encoder,
                 num_epochs,
                 learning_rate,
                 ):
        self.named_constants = ModelNamedConstant()
        # =============================Model hyper-parameters=============================
        self.vocabulary_size_encoder = vocabulary_size_encoder  # this includes one special tokens <KOMYPAD>=0 <KOMYEOS>=1 .........0 >TO> 98
        self.vocabulary_size_decoder = vocabulary_size_decoder  # this includes 2 special tokens <KOMYPAD>=0 <KOMYEOS>=1 <KOMYSOS>=2  ..........0 >TO> 99

        self.max_seq_len_encoder = max_seq_len_encoder
        self.max_seq_len_decoder = max_seq_len_decoder

        self.num_samples_train = num_samples_train
        self.num_samples_test = num_samples_test
        self.num_buckets_train = num_buckets_train
        self.num_buckets_test = num_buckets_test
        self.batch_size = batch_size

        assert self.num_samples_train % self.batch_size == 0
        assert self.num_samples_test % self.batch_size == 0

        assert (self.num_samples_train // self.num_buckets_train) % self.batch_size == 0
        assert (self.num_samples_test // self.num_buckets_test) % self.batch_size == 0

        self.use_embedding = use_embedding
        self.encoder_embedding_size = encoder_embedding_size
        self.decoder_embedding_size = decoder_embedding_size

        self.stacked_layers = stacked_layers
        self.keep_prop = keep_prop
        self.internal_state_encoder = internal_state_encoder
        self.internal_state_decoder = self.internal_state_encoder * 2

        self.num_epochs = num_epochs

        self.learning_rate = learning_rate
