# OpenNLMTK

This repo contains a neural network language modeling toolkit. Currently, it just provides a RNNLM with a small number of figures. Please, feel free to contribute to improve this repo.

## Usage:

```
$ python train.py
usage: train.py [-h] -model MODEL -train TRAIN -valid VALID [-gpu_id GPU_ID]
                [-batch_size BATCH_SIZE] [-n_epochs N_EPOCHS]
                [-print_every PRINT_EVERY] [-early_stop EARLY_STOP]
                [-iter_ratio_in_epoch ITER_RATIO_IN_EPOCH] [-dropout DROPOUT]
                [-word_vec_dim WORD_VEC_DIM] [-hidden_size HIDDEN_SIZE]
                [-max_length MAX_LENGTH] [-n_layers N_LAYERS]
                [-max_grad_norm MAX_GRAD_NORM] [-lr LR] [-min_lr MIN_LR]
```

or you can refer the example shell script file to take several experiments.


## Evaluation:

Below is a list of model files from the training. These filenames contain experiment setting, epochs, training loss, training PPL, validation loss and validation PPL.
We can figure out that this simple RNNLM (using LSTM) architecture is much better than n-gram in Appendix. I strongly believe that hyper-parameter tuning would improve performance of this language model, also. 

You may need to try it. :)

- joongang_daily corpus PPL: 369.34
- TED corpus PPL: 303.35
- joongang_daily + TED corpus PPL: 303.87

```
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:31 en.sgd.iter-.1.01.6.03-416.14.6.08-436.64.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:32 en.sgd.iter-.1.02.5.47-236.45.5.81-334.33.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:32 en.sgd.iter-.1.03.5.35-210.04.5.82-338.41.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:32 en.sgd.iter-.1.04.5.00-148.43.5.76-317.09.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:33 en.sgd.iter-.1.05.5.03-153.69.5.72-305.10.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:33 en.sgd.iter-.1.06.5.09-162.35.5.72-304.84.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:33 en.sgd.iter-.1.07.5.01-149.91.5.72-305.51.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:33 en.sgd.iter-.1.08.4.97-144.28.5.74-310.16.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:34 en.sgd.iter-.1.09.4.98-144.91.5.73-308.34.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:34 en.sgd.iter-.1.10.5.01-149.40.5.73-308.27.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:40 en.sgd.iter-.1.dropout-.3.01.6.03-413.77.6.19-485.43.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:41 en.sgd.iter-.1.dropout-.3.02.5.58-265.40.5.77-321.04.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:41 en.sgd.iter-.1.dropout-.3.03.5.25-190.68.5.88-356.51.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:41 en.sgd.iter-.1.dropout-.3.04.5.19-179.53.5.73-308.86.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:41 en.sgd.iter-.1.dropout-.3.05.5.17-175.68.5.73-307.63.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:42 en.sgd.iter-.1.dropout-.3.06.5.07-158.97.5.75-312.80.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:42 en.sgd.iter-.1.dropout-.3.07.5.08-160.53.5.72-304.53.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:42 en.sgd.iter-.1.dropout-.3.08.5.07-159.37.5.72-305.34.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:43 en.sgd.iter-.1.dropout-.3.09.5.01-149.74.5.72-304.24.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:43 en.sgd.iter-.1.dropout-.3.10.5.01-150.51.5.72-304.21.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:43 en.sgd.iter-.1.dropout-.3.11.5.02-151.57.5.72-304.04.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:43 en.sgd.iter-.1.dropout-.3.12.5.11-164.94.5.72-303.87.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:44 en.sgd.iter-.1.dropout-.3.13.5.05-155.63.5.72-303.99.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:44 en.sgd.iter-.1.dropout-.3.14.5.08-160.96.5.72-303.95.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:44 en.sgd.iter-.1.dropout-.3.15.5.01-150.45.5.72-303.96.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:45 en.sgd.iter-.1.dropout-.3.16.5.13-169.52.5.72-303.96.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:21 en.sgd.iter-.2.01.5.57-261.50.5.97-389.99.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:22 en.sgd.iter-.2.02.5.16-174.77.5.81-332.96.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:22 en.sgd.iter-.2.03.4.99-146.26.5.85-346.53.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:23 en.sgd.iter-.2.04.4.79-119.95.5.83-340.18.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:23 en.sgd.iter-.2.05.4.74-114.62.5.81-335.22.pth
-rw-rw-r-- 1 khkim khkim  70M  5월 15 10:24 en.sgd.iter-.2.06.4.68-107.30.5.82-335.64.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:36 joongang_daily.en.sgd.iter-.1.01.7.54-1889.45.7.16-1282.53.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:37 joongang_daily.en.sgd.iter-.1.02.6.99-1083.39.6.72-827.21.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:37 joongang_daily.en.sgd.iter-.1.03.6.70-814.76.6.45-631.64.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:37 joongang_daily.en.sgd.iter-.1.04.6.43-622.39.6.21-497.04.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:37 joongang_daily.en.sgd.iter-.1.05.6.17-478.78.6.17-476.52.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:37 joongang_daily.en.sgd.iter-.1.06.6.08-436.09.5.98-393.90.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:37 joongang_daily.en.sgd.iter-.1.07.5.93-375.82.6.14-464.74.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:37 joongang_daily.en.sgd.iter-.1.08.5.75-314.15.6.06-428.98.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:37 joongang_daily.en.sgd.iter-.1.09.5.64-280.24.6.04-418.73.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:37 joongang_daily.en.sgd.iter-.1.10.5.62-275.78.6.04-417.93.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:28 joongang_daily.en.sgd.iter-.2.01.6.90-995.23.6.64-762.23.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:28 joongang_daily.en.sgd.iter-.2.02.6.35-574.93.6.46-641.16.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:29 joongang_daily.en.sgd.iter-.2.03.6.01-406.78.6.09-443.06.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:29 joongang_daily.en.sgd.iter-.2.04.5.78-322.38.5.91-369.34.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:29 joongang_daily.en.sgd.iter-.2.05.5.54-255.53.6.02-410.61.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:29 joongang_daily.en.sgd.iter-.2.06.5.40-221.96.5.95-384.39.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:29 joongang_daily.en.sgd.iter-.2.07.5.30-200.02.5.96-387.91.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:29 joongang_daily.en.sgd.iter-.2.08.5.28-197.07.5.96-387.33.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:48 joongang_daily.en.sgd.iter-.2.dropout-.3.01.6.94-1031.00.6.49-659.06.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:48 joongang_daily.en.sgd.iter-.2.dropout-.3.02.6.29-539.51.6.16-472.09.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:48 joongang_daily.en.sgd.iter-.2.dropout-.3.03.6.05-422.37.6.13-459.78.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:48 joongang_daily.en.sgd.iter-.2.dropout-.3.04.5.88-357.84.6.00-402.30.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:48 joongang_daily.en.sgd.iter-.2.dropout-.3.05.5.68-293.43.6.00-401.64.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:49 joongang_daily.en.sgd.iter-.2.dropout-.3.06.5.53-252.49.6.09-440.99.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:49 joongang_daily.en.sgd.iter-.2.dropout-.3.07.5.32-204.21.5.98-394.34.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:49 joongang_daily.en.sgd.iter-.2.dropout-.3.08.5.36-213.50.5.95-383.33.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:49 joongang_daily.en.sgd.iter-.2.dropout-.3.09.5.26-192.93.6.01-406.71.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:49 joongang_daily.en.sgd.iter-.2.dropout-.3.10.5.30-199.76.5.97-392.97.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:49 joongang_daily.en.sgd.iter-.2.dropout-.3.11.5.28-195.43.5.97-393.13.pth
-rw-rw-r-- 1 khkim khkim  60M  5월 15 10:49 joongang_daily.en.sgd.iter-.2.dropout-.3.12.5.28-196.59.5.97-393.13.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:34 ted.en.sgd.iter-.1.01.5.87-354.67.6.18-481.23.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:34 ted.en.sgd.iter-.1.02.5.35-209.59.6.22-501.48.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:35 ted.en.sgd.iter-.1.03.5.11-165.58.5.82-337.29.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:35 ted.en.sgd.iter-.1.04.5.06-157.39.5.81-332.40.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:35 ted.en.sgd.iter-.1.05.4.97-143.86.5.77-321.58.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:35 ted.en.sgd.iter-.1.06.4.99-146.84.5.75-313.51.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:35 ted.en.sgd.iter-.1.07.4.99-147.61.5.74-311.03.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:36 ted.en.sgd.iter-.1.08.4.99-146.48.5.71-303.35.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:36 ted.en.sgd.iter-.1.09.4.82-123.44.5.74-312.02.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:36 ted.en.sgd.iter-.1.10.4.81-122.64.5.72-305.80.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:36 ted.en.sgd.iter-.1.11.4.71-110.71.5.73-307.65.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:36 ted.en.sgd.iter-.1.12.4.91-136.01.5.73-307.61.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:45 ted.en.sgd.iter-.1.dropout-.3.01.6.01-407.95.6.14-462.55.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:45 ted.en.sgd.iter-.1.dropout-.3.02.5.45-232.43.6.00-402.03.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:45 ted.en.sgd.iter-.1.dropout-.3.03.5.29-198.39.5.83-339.97.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:45 ted.en.sgd.iter-.1.dropout-.3.04.5.01-149.23.5.78-324.47.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:46 ted.en.sgd.iter-.1.dropout-.3.05.4.93-137.76.5.83-341.11.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:46 ted.en.sgd.iter-.1.dropout-.3.06.4.78-119.63.5.75-314.91.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:46 ted.en.sgd.iter-.1.dropout-.3.07.4.81-122.77.5.76-316.39.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:46 ted.en.sgd.iter-.1.dropout-.3.08.4.81-122.14.5.75-312.93.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:46 ted.en.sgd.iter-.1.dropout-.3.09.4.77-118.30.5.74-311.93.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:47 ted.en.sgd.iter-.1.dropout-.3.10.4.80-121.11.5.74-311.67.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:47 ted.en.sgd.iter-.1.dropout-.3.11.4.74-114.68.5.74-310.77.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:47 ted.en.sgd.iter-.1.dropout-.3.12.4.76-117.21.5.74-310.71.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:47 ted.en.sgd.iter-.1.dropout-.3.13.4.74-114.36.5.74-311.35.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:47 ted.en.sgd.iter-.1.dropout-.3.14.4.69-108.74.5.74-310.73.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:47 ted.en.sgd.iter-.1.dropout-.3.15.4.79-120.30.5.74-310.71.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:48 ted.en.sgd.iter-.1.dropout-.3.16.4.76-116.66.5.74-310.71.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:24 ted.en.sgd.iter-.2.01.5.34-208.65.6.13-461.57.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:25 ted.en.sgd.iter-.2.02.4.99-146.76.5.92-373.73.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:25 ted.en.sgd.iter-.2.03.4.77-117.37.5.80-329.21.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:25 ted.en.sgd.iter-.2.04.4.68-107.58.5.82-337.95.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:26 ted.en.sgd.iter-.2.05.4.52-91.86.5.80-329.75.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:26 ted.en.sgd.iter-.2.06.4.48-88.63.5.78-323.54.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:27 ted.en.sgd.iter-.2.07.4.48-88.33.5.77-322.14.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:27 ted.en.sgd.iter-.2.08.4.48-88.13.5.78-322.22.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:27 ted.en.sgd.iter-.2.09.4.46-86.67.5.78-322.76.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:28 ted.en.sgd.iter-.2.10.4.51-91.20.5.78-322.81.pth
-rw-rw-r-- 1 khkim khkim  63M  5월 15 10:28 ted.en.sgd.iter-.2.11.4.43-83.98.5.78-322.81.pth
```

## Appendix:

Below is evaluation results from n-gram language modeling with same training data and test data.

joongang_daily corpus PPL: 510.9258
```
$ time ngram-count -order 3 -kndiscount -text ./data/joongang_daily.aligned.en.refined.tok.bpe.txt -lm ./data/joongang_daily.aligned.en.refined.tok.bpe.lm -write-vocab ./data/joongang_daily.aligned.en.refined.tok.bpe.vocab.txt -debug 2
$ ngram -ppl ./data/test.refined.tok.bpe.txt -lm ./data/joongang_daily.aligned.en.refined.tok.bpe.lm -order 3 -debug 2

file ./data/test.refined.tok.bpe.txt: 1000 sentences, 13302 words, 44 OOVs 0 zeroprobs, logprob= -38615.77 ppl= 510.9258 ppl1= 817.7845
```

TED corpus PPL: 374.1577
```
$ time ngram-count -order 3 -kndiscount -text ./data/ted.aligned.en.refined.tok.bpe.txt -lm ./data/ted.aligned.en.refined.tok.bpe.lm -write-vocab ./data/ted.agliend.en.refined.tok.bpe.vocab.txt -debug 2
$ ngram -ppl ./data/test.refined.tok.bpe.txt -lm ./data/ted.aligned.en.refined.tok.bpe.lm -order 3 -debug 2

file ./data/test.refined.tok.bpe.txt: 1000 sentences, 13302 words, 32 OOVs 0 zeroprobs, logprob= -36717.49 ppl= 374.1577 ppl1= 584.7292
```

joongang_daily(0.5) + TED(0.5) corpus PPL: 328.6022
```
$ ngram -lm ./data/joongang_daily.aligned.en.refined.tok.bpe.lm -mix-lm ./data/ted.aligned.en.refined.tok.bpe.lm -lambda .5 -write-lm ./data/joongang_daily_ted.aligned.en.refined.tok.bpe.lm -debug 2
$ ngram -ppl ./data/test.refined.tok.bpe.txt -lm ./data/joongang_daily_ted.aligned.en.refined.tok.bpe.lm -order 3 -debug 2

file ./data/test.refined.tok.bpe.txt: 1000 sentences, 13302 words, 18 OOVs 0 zeroprobs, logprob= -35948.12 ppl= 328.6022 ppl1= 508.3018
```
