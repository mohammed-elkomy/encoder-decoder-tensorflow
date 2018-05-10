# Neural machine translation - tensorflow

Encoder-decoder architecture as seq2seq model used for neural machine translation where the source sentence in compressed in thought vector dimensional space and then projected on the target vocabulary space and sampling at each step 

My dataset is synthetic dataset where the source sentence for example **1 2 3 4 <KOMYEOS>** the target translation would be **4 3 2 1 <KOMYEOS>** so it will learning to map source to destination language and has the ability to determine the end of the sequence using the **<KOMYEOS>** .. I used greedy sampling it made better results but after reaching low perplexity nearly 1 both greedy and probabilistic sampling would work 
 
I implemented both batching and bucketing for training efficiency 

![alt text](https://www.blognone.com/sites/default/files/externals/457b91b1c143ae37eacf1f7b930e104e.jpg)

## Some sample logs while training 

```
train data: 4194304 samples,each bucket has 262144 samples , each bucket has 1024 batches
test data: 327680 samples,each bucket has 81920 samples , each bucket has 320 batches
Hyper parameters
+-------------------------+---------+
|        Parameter        |  Value  |
+-------------------------+---------+
| vocabulary_size_encoder |    69   |
| vocabulary_size_decoder |    70   |
|   max_seq_len_encoder   |    48   |
|   max_seq_len_decoder   |    52   |
|    num_samples_train    | 4194304 |
|     num_samples_test    |  327680 |
|    num_buckets_train    |    16   |
|     num_buckets_test    |    4    |
|        batch_size       |   256   |
|  encoder_embedding_size |    64   |
|  decoder_embedding_size |    64   |
|      stacked_layers     |    2    |
|        keep_prop        |   0.2   |
|  internal_state_encoder |   512   |
|  internal_state_decoder |   1024  |
|        num_epochs       |    10   |
|      learning_rate      |  0.001  |
+-------------------------+---------+
00:03:01.20 : (Training) step :217, epoch :0,avg minibatch-accuracy :0.0279 ,last mini-batch loss:4.0832 ,last mini-batch perplexity:59.7437
Example generation
11 60 55 48 26 2 54 64 63 56 15 4 42 10 62 <KOMYEOS>  ==> 56 10 10 56 56 10 68 68 48 48 46 66 23 <KOMYPAD> <KOMYPAD> <KOMYPAD> <KOMYPAD> <KOMYPAD> <KOMYPAD> <KOMYPAD> <KOMYPAD> (1/16) .
11 60 55 48 26 2 54 64 63 56 15 4 42 10 62 1  ==> 56 10 10 56 56 10 68 68 48 48 46 66 23 0 0 0 0 0 0 0 0 (1/16) .

............ LATER ....... 
28 8 55 19 47 49 55 35 62 25 63 24 58 27 5 41 28 16 49 26 44 <KOMYEOS>  ==> 44 26 49 16 28 41 5 27 58 63 24 25 62 35 49 55 47 19 55 8 28 19 <KOMYPAD> 48 <KOMYPAD> 6 46 (17/22) .
28 8 55 19 47 49 55 35 62 25 63 24 58 27 5 41 28 16 49 26 44 1  ==> 44 26 49 16 28 41 5 27 58 63 24 25 62 35 49 55 47 19 55 8 28 19 0 48 0 6 46 (17/22) .

............ LATER ....... 

52 27 44 7 17 65 40 32 24 18 61 28 34 43 67 68 53 33 30 21 36 <KOMYEOS>  ==> 36 21 30 33 53 68 67 43 34 28 61 18 24 32 40 65 17 7 44 27 52 <KOMYEOS> <KOMYPAD> 54 <KOMYPAD> <KOMYPAD> 6 (22/22) .
52 27 44 7 17 65 40 32 24 18 61 28 34 43 67 68 53 33 30 21 36 1  ==> 36 21 30 33 53 68 67 43 34 28 61 18 24 32 40 65 17 7 44 27 52 0 0 54 1 0 6 (22/22) .

```

I sample with 2 different techniques(to show they are equivalent) that's why I have the same sequence repeated 

## Resources

* [TensorFlow tutorials](https://www.tensorflow.org/tutorials/seq2seq)
* [R2RT blog post](https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)
* [Siraj video](https://www.youtube.com/watch?v=ElmBrKyMXxs) But he didn't really implement attention mechanism
* [good video tutorial](https://www.youtube.com/watch?v=_Sm0q_FckM8)


