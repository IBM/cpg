import argparse

import numpy as np
import torch
from torchtext import data, datasets

from src.scan.model import SCANModel

from src.scan.data import load_SCAN_length, load_SCAN_simple, build_vocab, preprocess, MyDataLoader


def evaluate(args):
    # load test data
    _, test_data = load_SCAN_length()
    x_vocab = build_vocab([x for x, _ in test_data],
                           base_tokens=['<PAD>', '<UNK>'])
    y_vocab = build_vocab([y for _, y in test_data],
                           base_tokens=['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
    preprocessed_test_data = preprocess(test_data, x_vocab, y_vocab)
    test_loader = MyDataLoader(preprocessed_test_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                                y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                                max_x_seq_len=args.max_x_seq_len,
                                max_y_seq_len=args.max_y_seq_len)
    
    model = SCANModel(y_vocab=y_vocab, x_vocab=x_vocab,
                     word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                     decoder_hidden_dim=args.decoder_hidden_dim,
                     decoder_num_layers=args.decoder_num_layers,
                     use_leaf_rnn=args.leaf_rnn,
                     bidirectional=args.bidirectional,
                     intra_attention=args.intra_attention,
                     use_batchnorm=args.batchnorm,
                     dropout_prob=args.dropout,
                     max_y_seq_len=args.max_y_seq_len)
    
    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.embedding.weight.size())
    print(f'# of parameters: {num_params}')
    print(f'# of word embedding parameters: {num_embedding_params}')
    print(f'# of parameters (excluding word embeddings): '
          f'{num_params - num_embedding_params}')
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    model.to(args.device)
    torch.set_grad_enabled(False)
    accuracy_sum = 0
    for batch in test_loader:
        batch_x, batch_y = batch
        B, L = batch_x.size()
        outputs, _, _ = model(x=batch_x, length=torch.full((B, 1), L).view(B), force=batch_y)
        _, M = batch_y.size()
        outputs = outputs[:, :M]
        outputs = outputs.reshape(B * M)
        accuracy = torch.eq(batch_y.view(B * M), outputs).float().mean()
        accuracy_sum += accuracy.item()
    accuracy = accuracy_sum / test_loader.num_batches
    print(f'accuracy: {accuracy:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--hidden-dim', required=True, type=int)
    parser.add_argument('--decoder-hidden-dim', required=True, type=int)
    parser.add_argument('--decoder-num-layers', required=True, type=int)
    parser.add_argument('--leaf-rnn', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--fine-grained', default=False, action='store_true')
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--max_x_seq_len', default = 9)
    parser.add_argument('--max_y_seq_len', default = 49)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
