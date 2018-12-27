# Neural Machine Translation - tensorflow

Encoder-decoder architecture as seq2seq model used for neural machine translation where the source sentence in compressed in thought vector dimensional space and then projected on the target vocabulary space and sampling at each step 

My dataset is synthetic dataset where the source sentence for example **1 2 3 4 </KOMYEOS/>** the target translation would be **4 3 2 1 </KOMYEOS/>** so it will learning to map source to destination language and has the ability to determine the end of the sequence using the **</KOMYEOS/>** .. I used greedy sampling it made better results but after reaching low perplexity nearly 1 both greedy and probabilistic sampling would work 
 
I implemented both batching and bucketing for training efficiency 

![NMT model](https://www.blognone.com/sites/default/files/externals/457b91b1c143ae37eacf1f7b930e104e.jpg)

## How to run demo
  ```
  # vanilla demo
  cd vanilla
  python demo.py
  
  # attention model
  cd atten_tf
  python demo.py
  ```
## Demo output(my pretrained models)
### vanila demo
```
Enter Statement
please invert this sentence vanilla seq2seq model? <KOMYEOS> 
==>
?ledom qes2qes allinav ecnetnes siht trevni esaelp <KOMYEOS>(51/51) .
```
### attentive non-inverting model demo + forward attention grid
```
Enter Statement
show me how you really attend? <KOMYEOS> ==>
show me how you really attend? <KOMYEOS>(31/31) .
```
![Demo attention](https://serving.photos.photobox.com/56163462d87c4a51773d73757f6534d17e2b342d470c26b525f3e4a77cb3fb6dc0864635.jpg)

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
Average padding / batch: 334.575 as the batch has 12288 tokens padding percentage: 0.027227783203124998
00:03:00.30 : (Training) step :281, epoch :0,avg minibatch-accuracy :0.0585 ,last mini-batch loss:3.5171 ,last mini-batch perplexity:35.4528
Example generation
9 38 30 16 31 37 26 25 61 11 61 29 5 25 58 41 62 48 45 38 26 29 9 57 16 61 66 <KOMYEOS>  ==> 16 29 29 29 29 29 29 61 37 37 41 37 58 68 61 61 33 42 31 31 9 58 67 61 <KOMYEOS> <KOMYEOS> <KOMYEOS> 26 <KOMYEOS> <KOMYEOS> 60 51 10 (2/28) .
**************************************************
43 12 27 40 52 58 36 62 10 21 6 41 17 52 16 11 56 66 11 61 16 50 <KOMYEOS>  ==> 11 61 61 11 66 66 66 16 36 62 6 6 62 21 21 28 41 2 56 10 50 62 62 46 34 42 55 13 (5/23) .
**************************************************
52 28 25 36 60 11 33 26 11 25 54 68 26 23 <KOMYEOS>  ==> 68 23 11 11 11 25 25 36 36 11 60 37 33 60 33 <KOMYEOS> <KOMYEOS> <KOMYEOS> <KOMYEOS> <KOMYEOS> (0/15) .
**************************************************
................later..............
**************************************************
00:27:03.08 : (Training) step :265, epoch :0,avg minibatch-accuracy :0.7344 ,last mini-batch loss:1.0155 ,last mini-batch perplexity:3.2849
Example generation
56 50 59 53 8 25 29 64 40 66 47 36 5 63 68 63 64 59 5 6 26 <KOMYEOS>  ==> 26 6 5 59 64 63 68 63 5 36 47 66 64 40 29 25 8 53 50 59 56 <KOMYEOS> <KOMYEOS> <KOMYEOS> 24 <KOMYEOS> <KOMYEOS> (18/22) .
**************************************************
23 52 65 20 64 31 9 60 67 21 64 64 60 58 66 42 17 34 60 58 30 30 13 <KOMYEOS>  ==> 13 30 30 58 60 34 17 42 66 58 60 64 64 21 67 45 9 60 64 20 65 23 52 <KOMYEOS> <KOMYEOS> 60 <KOMYEOS> <KOMYEOS> 24 (20/24) .
**************************************************
64 57 3 55 11 46 25 65 39 10 14 56 29 20 9 6 41 9 66 8 <KOMYEOS>  ==> 8 66 9 41 6 9 29 20 56 14 10 39 65 25 46 11 55 3 57 64 <KOMYEOS> <KOMYEOS> 10 <KOMYEOS> <KOMYEOS> 11 (19/21) .
**************************************************
................later..............
**************************************************
00:36:10.88 : (Training) step :278, epoch :0,avg minibatch-accuracy :0.8047 ,last mini-batch loss:0.3251 ,last mini-batch perplexity:2.0926
Example generation
52 18 57 19 50 29 12 38 49 50 33 39 27 6 37 29 67 44 45 39 46 21 3 21 <KOMYEOS>  ==> 21 3 21 46 39 45 67 44 29 37 6 27 39 33 50 49 38 29 12 50 19 57 18 52 <KOMYEOS> <KOMYEOS> <KOMYEOS> 24 <KOMYEOS> <KOMYEOS> (21/25) .
52 18 57 19 50 29 12 38 49 50 33 39 27 6 37 29 67 44 45 39 46 21 3 21 1  ==> 21 3 21 46 39 45 67 44 29 37 6 27 39 33 50 49 38 29 12 50 19 57 18 52 <KOMYEOS> <KOMYEOS> <KOMYEOS> 24 <KOMYEOS> <KOMYEOS> (21/25) .
**************************************************
55 45 19 25 66 27 23 64 65 32 32 50 52 56 48 <KOMYEOS>  ==> 48 56 52 50 32 32 65 64 23 27 66 25 19 45 55 <KOMYEOS> <KOMYEOS> 24 <KOMYEOS> 11 <KOMYEOS> (16/16) .
55 45 19 25 66 27 23 64 65 32 32 50 52 56 48 1  ==> 48 56 52 50 32 32 65 64 23 27 66 25 19 45 55 <KOMYEOS> <KOMYEOS> 24 <KOMYEOS> 11 <KOMYEOS> (16/16) .
**************************************************
6 29 11 40 53 22 7 17 27 55 66 43 55 33 33 30 6 50 33 27 19 35 28 <KOMYEOS>  ==> 28 35 19 27 33 50 6 30 33 33 55 66 43 55 27 17 7 22 53 40 11 29 6 <KOMYEOS> <KOMYEOS> <KOMYEOS> <KOMYEOS> 11 <KOMYEOS> (22/24) .
6 29 11 40 53 22 7 17 27 55 66 43 55 33 33 30 6 50 33 27 19 35 28 1  ==> 28 35 19 27 33 50 6 30 33 33 55 66 43 55 27 17 7 22 53 40 11 29 6 <KOMYEOS> <KOMYEOS> <KOMYEOS> <KOMYEOS> 11 <KOMYEOS> (22/24) .
**************************************************
................later..............
**************************************************
02:24:23.25 : (Training) step :271, epoch :0,avg minibatch-accuracy :0.9555 ,last mini-batch loss:0.0968 ,last mini-batch perplexity:1.5303
Example generation
51 68 27 21 59 10 59 46 53 45 59 14 65 61 65 28 46 25 10 8 <KOMYEOS>  ==> 8 10 25 46 28 65 61 65 14 59 45 53 46 59 10 59 21 27 68 51 <KOMYEOS> <KOMYEOS> 60 26 <KOMYEOS> 51 (21/21) .
**************************************************
57 14 14 51 45 11 31 26 46 13 52 25 42 51 <KOMYEOS>  ==> 51 42 25 52 13 46 26 31 11 45 51 14 14 57 <KOMYEOS> <KOMYEOS> 60 26 <KOMYEOS> 14 (15/15) .
**************************************************
................later..............
**************************************************
03:16:49.17 : (Training) step :266, epoch :1,avg minibatch-accuracy :0.9703 ,last mini-batch loss:0.0049 ,last mini-batch perplexity:2.2238
Example generation
**************************************************
67 46 32 39 44 3 17 37 17 62 28 27 57 17 50 <KOMYEOS>  ==> 50 17 57 27 28 62 17 37 17 3 44 39 32 46 67 <KOMYEOS> <KOMYEOS> 10 <KOMYEOS> 24 60 (16/16) .
**************************************************
27 32 44 51 53 26 46 7 22 30 56 58 22 15 9 66 <KOMYEOS>  ==> 66 9 15 22 58 56 30 22 7 46 26 53 51 44 32 27 <KOMYEOS> <KOMYEOS> <KOMYEOS> 10 10 24 (17/17) .
**************************************************
12 57 14 23 7 34 14 49 22 35 6 6 62 60 35 68 56 48 16 15 35 15 32 <KOMYEOS>  ==> 32 15 35 15 16 48 56 68 35 60 62 6 6 35 22 49 14 34 7 23 14 57 12 <KOMYEOS> <KOMYEOS> <KOMYEOS> 60 10 <KOMYEOS> (24/24) .
**************************************************

```

I sample with 2 different techniques(to show they are equivalent) that's why I have the same sequence repeated 

## Resources

* [TensorFlow tutorials](https://www.tensorflow.org/tutorials/seq2seq)
* [R2RT blog post](https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)
* [Siraj video](https://www.youtube.com/watch?v=ElmBrKyMXxs) But he didn't really implement attention mechanism
* [good video tutorial](https://www.youtube.com/watch?v=_Sm0q_FckM8)


