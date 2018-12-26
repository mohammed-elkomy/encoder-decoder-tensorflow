import sys

import tensorflow as tf
from tensorflow.python.client import device_lib

from model_configs import ModelConfigs
from model_graph import ModelGraphPureAtt
from training_manager import TrainingManager
from ..data_generator import DataGenerator
from ..logger import Logger

print(device_lib.list_local_devices())
print(tf.__version__)
print(sys.version)

project_name = "nmt_atten_non_inverting2"

batch = 256
buck_train = 16
buck_test = 4
samples_test = (50000 // (batch*buck_test)) * (batch*buck_test)
samples_train = (500000 // (batch*buck_train)) * (batch*buck_train)

modelConfigs = ModelConfigs(
    vocabulary_size_encoder=99,  # this includes one special tokens <KOMYPAD>=0 <KOMYEOS>=1 .........0 >TO> 98
    vocabulary_size_decoder=100,  # this includes 2 special tokens <KOMYPAD>=0 <KOMYEOS>=1 <KOMYSOS>=2  ..........0 >TO> 99
    max_seq_len_encoder=96,
    max_seq_len_decoder=108,
    num_samples_train=samples_train,
    num_samples_test=samples_test,
    num_buckets_train=16,
    num_buckets_test=4,
    batch_size=batch,
    use_embedding=True,
    encoder_embedding_size=256,
    decoder_embedding_size=256,
    stacked_layers=2,
    keep_prop=.2,
    internal_state_encoder=256,
    num_epochs=1000,
    learning_rate=1e-4,  # 1e-3
)

trainingManager = TrainingManager(
    configs=modelConfigs,
    project_name=project_name,
    summary_every_mins=3,
    checkpoint_every_mins=30,  # 30
    example_every_mins=5,
    test_every_mins=45,
    resume_training=True,
)

logger = Logger(trainingManager)
modelGraph = ModelGraphPureAtt(trainingManager)
dataGenerator = DataGenerator(trainingManager)

# if not trainingManager.is_local_env:
#     dataGenerator.average_padding()  # evaluating bucketed padding

trainingManager.launch_training_loop()
