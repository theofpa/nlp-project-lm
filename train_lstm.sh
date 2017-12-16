# LSTM's
# Zaremba et al 2014

# Medium Regularized
python --model LSTM --epochs 39 --emsize 60 --batch-size 20 --nhid 650 --nlayers 1 --lr 1 --clip 5 bptt 35 --dropout 0.5 --save rnn_00 --seed 257

# Large Regularized
python --model LSTM --epochs 55 --batch-size 20 --emsize 60 --nhid 1500 --nlayers 1 --lr 1 --clip 5 --bptt 35 --dropout 0.5 --save rnn_00 --seed 257


