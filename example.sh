# Initial running
python ./train.py -model ./models/en.sgd.iter-.2.pth -train ./data/aligned.en.refined.tok.bpe.txt -valid ./data/test.refined.tok.bpe.txt -print_every 50 -gpu_id 0 -iter_ratio_in_epoch .2 -n_epochs 20
python ./train.py -model ./models/ted.en.sgd.iter-.2.pth -train ./data/ted.aligned.en.refined.tok.bpe.txt -valid ./data/test.refined.tok.bpe.txt -print_every 50 -gpu_id 0 -iter_ratio_in_epoch .2 -n_epochs 20
python ./train.py -model ./models/joongang_daily.en.sgd.iter-.2.pth -train ./data/joongang_daily.aligned.en.refined.tok.bpe.txt -valid ./data/test.refined.tok.bpe.txt -print_every 50 -gpu_id 0 -iter_ratio_in_epoch .2 -n_epochs 20

# Second trial: change -iter_ratio_in_epoch .2 --> .1
python ./train.py -model ./models/en.sgd.iter-.1.pth -train ./data/aligned.en.refined.tok.bpe.txt -valid ./data/test.refined.tok.bpe.txt -print_every 50 -gpu_id 0 -iter_ratio_in_epoch .1 -n_epochs 20
python ./train.py -model ./models/ted.en.sgd.iter-.1.pth -train ./data/ted.aligned.en.refined.tok.bpe.txt -valid ./data/test.refined.tok.bpe.txt -print_every 50 -gpu_id 0 -iter_ratio_in_epoch .1 -n_epochs 20
python ./train.py -model ./models/joongang_daily.en.sgd.iter-.1.pth -train ./data/joongang_daily.aligned.en.refined.tok.bpe.txt -valid ./data/test.refined.tok.bpe.txt -print_every 50 -gpu_id 0 -iter_ratio_in_epoch .1 -n_epochs 20

# 3rd trial: increase dropout rate from .1 to .3. It will help to regularize for difficult test-set.
python ./train.py -model ./models/en.sgd.iter-.1.dropout-.3.pth -train ./data/aligned.en.refined.tok.bpe.txt -valid ./data/test.refined.tok.bpe.txt -print_every 50 -gpu_id 0 -iter_ratio_in_epoch .1 -n_epochs 20 -dropout .3
python ./train.py -model ./models/ted.en.sgd.iter-.1.dropout-.3.pth -train ./data/ted.aligned.en.refined.tok.bpe.txt -valid ./data/test.refined.tok.bpe.txt -print_every 50 -gpu_id 0 -iter_ratio_in_epoch .1 -n_epochs 20 -dropout .3
python ./train.py -model ./models/joongang_daily.en.sgd.iter-.2.dropout-.3.pth -train ./data/joongang_daily.aligned.en.refined.tok.bpe.txt -valid ./data/test.refined.tok.bpe.txt -print_every 50 -gpu_id 0 -iter_ratio_in_epoch .2 -n_epochs 20 -dropout .3
