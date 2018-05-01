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