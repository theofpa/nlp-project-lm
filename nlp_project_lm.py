import time
import math
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver.create(
    url='mongodb://uva:uva@ds159676.mlab.com:59676/uva',
    db_name='uva'))

@ex.config
def my_config():
    dataset = './data_ptb'  # location of the data corpus
    model_type = 'LSTM'     # type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
    emsize = 200            # size of word embeddings
    nhid = 200              # number of hidden units per layer
    nlayers = 2             # number of layers
    lr = float(20)          # initial learning rate
    clip = float(0.25)      # gradient clipping
    epochs = 40             # upper epoch limit
    batch_size = 32         # batch size
    bptt = 35               # sequence length
    dropout = float(0.2)    # dropout applied to layers (0 = no dropout)
    seed = 257              # random seed
    tied = True             # tie the word embedding and softmax weights
    log_interval = 200      # report interval
    save = "model.pt"       # path to save the final model

@ex.automain
def my_main(dataset, model_type, emsize, nhid, nlayers, lr, clip, epochs, batch_size, bptt, dropout, seed, tied, log_interval, save, _run):

    # Initialize seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load data
    print("Loading data...")
    corpus = data.Corpus(dataset)
    pickle.dump(corpus.dictionary.word2idx, open(save+'w2i', 'wb'))


    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        data = data.cuda()
        return data

    eval_batch_size = 10
    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    ntokens = len(corpus.dictionary)
    global model
    model = model.RNNModel(model_type, ntokens, emsize, nhid, nlayers, dropout, tied)
    model.cuda()
    criterion = nn.CrossEntropyLoss()


    # Training
    def repackage_hidden(h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(repackage_hidden(v) for v in h)

    def get_batch(source, i, evaluation=False):
        seq_len = min(bptt, len(source) - 1 - i)
        data = Variable(source[i:i+seq_len], volatile=evaluation)
        target = Variable(source[i+1:i+1+seq_len].view(-1))
        return data, target

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size)
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, evaluation=True)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss[0] / len(data_source)


    def train():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.data

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss[0] / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, lr,
                    elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    best_val_loss = None
    try:
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            _run.log_scalar("validation.loss", val_loss, epoch)
            _run.log_scalar("validation.perplexity", math.exp(val_loss), epoch)

            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    return math.exp(test_loss)
