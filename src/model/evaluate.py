import argparse
import logging
import os
from random import randint
import copy

from lark import Lark
from tensorboardX import SummaryWriter

import torch
import numpy as np
from torch import nn, optim
from torch.optim import lr_scheduler, optimizer
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from src.model.model import CompositionalLearner
from src.model.data import build_vocab, preprocess, MyDataLoader
from src.model.scan_data import load_SCAN_length, load_SCAN_simple, \
    load_SCAN_add_jump_0, load_SCAN_add_jump_4, parse_scan, \
    load_SCAN_add_jump_0_no_jump_oversampling, load_SCAN_add_turn_left, load_SCAN
from src.model.cogs_data import load_COGS, parse_cogs, cogs_grammar

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def evaluate(args):
    dataset = args.dataset
    training_set = args.training_set
    test_set = args.test_set
    if dataset == "SCAN":
        # load train and test scan_data
        train_data, test_data = load_SCAN(training_set, test_set)
        max_x_seq_len = 9
        max_y_seq_len = 49
        hidden_type_dim = 21
    elif dataset == "COGS":
        train_data, test_data = load_COGS(training_set, test_set)
        max_x_seq_len = 20
        max_y_seq_len = 160
        hidden_type_dim = 102
    training_size = int(len(train_data) * args.data_frac)
    train_data = train_data[:training_size]
    logging.info(f"Train scan_data set size: {len(train_data)}")
    logging.info(f"Train scan_data sample:\n\tx: {train_data[0][0]}\n\ty: {train_data[0][1]}")
    logging.info(f"Test scan_data set size: {len(test_data)}")
    logging.info(f"Test scan_data sample:\n\tx: {test_data[0][0]}\n\ty: {test_data[0][1]}")
    x_vocab = build_vocab([x for x, _ in train_data + test_data],
                           base_tokens=['<PAD>', '<UNK>'])
    y_vocab = build_vocab([y for _, y in train_data + test_data],
                           base_tokens=['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
    if dataset == "COGS":
        y_vocab.add_token("y")
        for i in range(20):
            y_vocab.add_token(str(i))
    logging.info(f"X Vocab size: {len(x_vocab)}")
    logging.info(f"Y Vocab size: {len(y_vocab)}")

    preprocessed_test_data = preprocess(test_data, x_vocab, y_vocab)

    test_loader = MyDataLoader(preprocessed_test_data,
                                batch_size=args.batch_size,
                                shuffle=True,
                                x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                                y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                                max_x_seq_len=max_x_seq_len,
                                max_y_seq_len=max_y_seq_len)
    
    model = CompositionalLearner(model=args.model,
                                 y_vocab=y_vocab,
                                 x_vocab=x_vocab,
                                 word_dim=args.word_dim,
                                 hidden_value_dim=args.hidden_dim,
                                 hidden_type_dim=hidden_type_dim,
                                 decoder_hidden_dim=args.decoder_hidden_dim,
                                 decoder_num_layers=args.decoder_num_layers,
                                 use_leaf_rnn=args.leaf_rnn,
                                 bidirectional=args.bidirectional,
                                 intra_attention=args.intra_attention,
                                 use_batchnorm=args.batchnorm,
                                 dropout_prob=args.dropout,
                                 max_y_seq_len=max_y_seq_len,
                                 use_prim_type_oracle=args.use_prim_type_oracle,
                                 syntactic_supervision=args.syntactic_supervision,
                                 dataset=dataset,
                                 eval=True)

    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    model.to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    torch.set_grad_enabled(False)
    
    def run_iter(batch, is_training=False, verbose=False):
        model.train(is_training)
        batch_x, batch_y = batch
        B, L = batch_x.size()
        if args.syntactic_supervision:
            positions_force = []
            types_force = []
            spans_force = []
            x_tokens = x_vocab.decode_batch(batch_x.numpy(), batch_x != x_vocab.token_to_idx('<PAD>'))
            for i in range(len(x_tokens)):
                x_tokens[i] = " ".join(x_tokens[i])
            positions = types = spans = None
            for i in range(B):
                if dataset == 'SCAN':
                    positions, types, spans = parse_scan(x_tokens[i])
                elif dataset == 'COGS':
                    # create parser
                    parser = Lark(cogs_grammar, propagate_positions=True)
                    positions, types, spans = parse_cogs(parser, x_tokens[i])
                else:
                    assert(False)
                positions_force.append(positions)
                types_force.append(types)
                spans_force.append(spans)
        else:
            positions_force = None
            types_force = None
            spans_force = None
        if is_training and args.use_teacher_forcing:
            force = batch_y
        else:
            force = None
        decoding = model(x=batch_x, length=torch.full((B, 1), L).view(B),force=force,
                         positions_force=positions_force, types_force=types_force, spans_force=spans_force)
        decoding_idx = decoding.argmax(-1)
        if verbose:
            input = x_vocab.decode_batch(batch_x.numpy(), batch_x != x_vocab.token_to_idx('<PAD>'))
            expected = y_vocab.decode_batch(batch_y.numpy(), batch_y != y_vocab.token_to_idx('<PAD>'))
            decoded = y_vocab.decode_batch(decoding_idx.numpy(), decoding_idx != y_vocab.token_to_idx('<PAD>'))
            print("--------------------------------")
            for i in range(B):
                print("input: ", " ".join(input[i]), "\n")
                print("expected: ", " ".join(expected[i]), "\n")
                print("decoded: ", " ".join(decoded[i]))
                print("--------------------------------")
        _, N, V = decoding.size()
        _, M = batch_y.size()
        if N >= M:
            # pad expected
            expected_padded = torch.full((B, N), float(y_vocab.token_to_idx('<PAD>')))
            expected_padded[:, :M] = batch_y
            outputs_padded = decoding_idx
        else:
            # pad outputs
            expected_padded = batch_y
            outputs_padded = torch.full((B, M), float(y_vocab.token_to_idx('<PAD>')))
            outputs_padded[:, :N] = decoding_idx


        # measure accuracy
        match = torch.eq(expected_padded.float(), outputs_padded.float())
        match = [(match[i].sum() == match.size(1)).float() for i in range(match.size(0))]

        # compute cross entropy loss of the decodings against the ground truth
        loss = F.cross_entropy(torch.clamp(torch.flatten(decoding, start_dim=0, end_dim=1), min=1.0e-10, max=1.0),
                               expected_padded.flatten().long())
        accuracy = torch.tensor(match).mean()
        if is_training:
            # zero gradients
            for param in model.parameters():
                param.grad = None
            loss.backward()
            clip_grad_norm_(parameters=params, max_norm=5)
            optimizer.step()
        return loss, accuracy
    
    test_loss_sum = test_accuracy_sum = 0
    num_test_batches = test_loader.num_batches
    for batch in test_loader:
        test_loss, test_accuracy = run_iter(batch=batch, verbose=args.print_in_valid)
        test_loss_sum += test_loss.item()
        test_accuracy_sum += test_accuracy.item()
    test_loss = test_loss_sum / num_test_batches
    test_accuracy = test_accuracy_sum / num_test_batches
    print(f'test loss: {test_loss:.4f}')
    print(f'test accuracy: {test_accuracy:.4f}')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--hidden-dim', required=True, type=int)
    parser.add_argument('--decoder-hidden-dim', required=True, type=int)
    parser.add_argument('--decoder-num-layers', required=True, type=int)
    parser.add_argument('--leaf-rnn', default=False, action='store_true')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--l2reg', default=0.0, type=float)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--fix-word-embedding', default=False, action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--max-epoch', required=True, type=int)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--omit-prob', default=0.0, type=float)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--fine-grained', default=False, action='store_true')
    parser.add_argument('--halve-lr-every', default=2, type=int)
    parser.add_argument('--scan_data-frac', default = 1., type=float)
    parser.add_argument('--use_teacher_forcing', default=True, action='store_true')
    parser.add_argument('--model', default='tree-lstm', choices={'tree-lstm', 'lstm'})
    parser.add_argument('--use-prim-type-oracle', default=False, action='store_true')
    parser.add_argument('--print-in-valid', default=False, action='store_true')
    parser.add_argument('--use-curriculum', default=False, action='store_true')
    parser.add_argument('--syntactic-supervision', default=False, action='store_true')
    parser.add_argument('--data-frac', default=1.0, type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--training-set', type=str)
    parser.add_argument('--test-set', type=str)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
