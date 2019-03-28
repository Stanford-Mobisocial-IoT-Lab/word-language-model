import argparse
import time
import math
import torch
import torch.nn as nn

from torchtext.data import TabularDataset, Field, Iterator, BucketIterator

import data
from model import RNNModel

def split_tokenize(x):
    return x.split()

def main():
    # Add ckp
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='/input', # /input
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint to use')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='/output/model.pt', # /output
                        help='path to save the final model')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # Load checkpoint
    build_vocab = False
    if args.checkpoint != '' and os.path.exists(args.checkpoint):
        print(f'Loading field from {args.checkpoint}')
        save_dict = torch.load(args.checkpoint)
        field = save_dict['field']
        start_epoch = save_dict['start_epoch']
    else:
        save_dict = None
        field = Field(tokenize=split_tokenize, init_token='<init>')
        build_vocab = True
        start_epoch = 0

    ###############################################################################
    # Load data
    ###############################################################################

    train_data, val_data, test_data = TabularDataset.splits(path=args.data, train='train.txt', validation='valid.txt', test='test.txt',
                                                            format='tsv', fields=[('text', field)])
    print(train_data, len(train_data), val_data, len(val_data), test_data, len(test_data))
    if build_vocab:
        field.eos_token = '<eos>'
        field.build_vocab(train_data, val_data, min_freq=1000)
        field.eos_token = None
    eos_id = field.vocab.stoi['<eos>']
    pad_id = field.vocab.stoi[field.pad_token]

    train_iter = BucketIterator(train_data, args.batch_size, train=True, repeat=False, device='cuda:0' if args.cuda else 'cpu:0')
    val_iter = Iterator(val_data, args.batch_size, repeat=False, device='cuda:0' if args.cuda else 'cpu:0')
    test_iter = Iterator(test_data, args.batch_size, repeat=False, device='cuda:0' if args.cuda else 'cpu:0')
    print(train_iter, len(train_iter), val_iter, len(val_iter), test_iter, len(test_iter))

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(field.vocab)
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

    if save_dict is not None:
        model.load_state_dict(save_dict['model'])

    if args.cuda:
        model.cuda()
    else:
        model.cpu()
    print (model)

    if save_dict:
        opt = save_dict['optimizer']
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.checkpoint:
        torch.save(dict(field=field, model=model.state_dict(), optimizer=opt, start_epoch=start_epoch), args.checkpoint)

    ###############################################################################
    # Training code
    ###############################################################################

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    def make_target(text):
        batch_size = text.size()[1]
        eos_vector = torch.full((1, batch_size), eos_id, dtype=text.dtype, device='cuda:0' if args.cuda else 'cpu:0')
        target = torch.cat((text[1:], eos_vector), dim=0)
        return target

    def compute_loss(output, text):
        output_flat = output.view(-1, ntokens)
        target = make_target(text)
        target_flat = target.view(-1)

        return criterion(output_flat, target_flat)

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        with torch.no_grad():
            model.eval()
            total_loss = 0
            for batch in data_source:
                output, hidden = model(batch.text)
                loss = compute_loss(output, batch.text)

                total_loss += loss.item()
            return total_loss / len(data_source)


    def train():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        start_time = time.time()
        for i, batch in enumerate(train_iter):
            model.zero_grad()

            output, hidden = model(batch.text)
            target = make_target(batch.text)

            loss = compute_loss(output, batch.text)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            total_loss += loss.item()

            if i % args.log_interval == 0 and i > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, len(train_iter),
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    # Loop over epochs.
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_iter)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if args.checkpoint:
                    torch.save(dict(field=field, model=model.state_dict(), optimizer=opt, start_epoch=epoch), args.checkpoint)
                best_val_loss = val_loss
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    torch.save(dict(vocab=field.vocab.itos, model=model.state_dict(),
                    settings=dict(rnn_type=args.model, emsize=args.emsize, nhid=args.nhid, nlayers=args.nlayers)),
               args.save)

    # Load the best saved model.
    #with open(args.save, 'rb') as f:
    #    save_dict = torch.load(f)
    #    field = save_dict['field']
    #    if save_dict is not None:
    #        model.load_state_dict(save_dict['model'])
    #
    #    if args.cuda:
    #        model.cuda()
    #    else:
    #        model.cpu()

    # Run on test data.
    test_loss = evaluate(test_iter)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

if __name__ == '__main__':
    main()
