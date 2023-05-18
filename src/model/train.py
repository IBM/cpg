import argparse
import logging
import os
from random import randint

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


def train(args):
    dataset = args.dataset
    training_set = args.training_set
    dev_set = args.dev_set
    if dataset == "SCAN":
        # load train and test scan_data
        train_data, test_data = load_SCAN(training_set, dev_set)
        max_x_seq_len = 9
        max_y_seq_len = 49
        hidden_type_dim = 21
    elif dataset == "COGS":
        train_data, test_data = load_COGS(training_set, dev_set)
        max_x_seq_len = 20
        max_y_seq_len = 160
        hidden_type_dim = 101
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

    curriculum_stage = None
    if args.use_curriculum:
        if dataset == "SCAN":
            curriculum_stage = 1
            filter_fn = lambda x: len(x[0]) == curriculum_stage
        elif dataset == "COGS":
            curriculum_stage = 3
            filter_fn = lambda x: 1 < len(x[0]) <= curriculum_stage
        train_data_curriculum = list(filter(filter_fn, train_data))
        #print(train_data_curriculum)
        preprocessed_train_data = preprocess(train_data_curriculum, x_vocab, y_vocab)
        valid_data_curriculum = list(filter(filter_fn, test_data))
        #print(valid_data_curriculum)
        preprocessed_valid_data = preprocess(valid_data_curriculum, x_vocab, y_vocab)

    else:
        preprocessed_train_data = preprocess(train_data, x_vocab, y_vocab)
        preprocessed_valid_data = preprocess(test_data, x_vocab, y_vocab)
    train_loader = MyDataLoader(preprocessed_train_data,
                                batch_size=args.batch_size,
                                shuffle=True,
                                x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                                y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                                max_x_seq_len=max_x_seq_len,
                                max_y_seq_len=max_y_seq_len)

    valid_loader = MyDataLoader(preprocessed_valid_data,
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
                                 dataset=dataset)
    if args.pretrained:
        model.embedding.weight.data.set_(y_vocab.vectors)
    if args.fix_word_embedding:
        logging.info('Will not update word embeddings')
        model.embedding.weight.requires_grad = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {args.device}')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    optimizer = optimizer_class(params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5,
        patience=20 * args.halve_lr_every, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'train'))
    valid_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'valid'))

    def run_iter(batch, average_accuracy, is_training=False, verbose=False):
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
        loss = F.cross_entropy(torch.clamp(torch.flatten(decoding, start_dim=0, end_dim=1), min=1.0e-20, max=1.0),
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

    def add_scalar_summary(summary_writer, name, value, step):
        if torch.is_tensor(value):
            value = value.item()
        summary_writer.add_scalar(tag=name, scalar_value=value, global_step=step)

    best_vaild_accuacy = 0
    iter_count = 1
    train_accuracy_stage_list = []
    train_accuracy_stage = 0.0
    iter_count_stage = 1
    validated = False
    for e in range(args.max_epoch):
        if args.use_curriculum and train_accuracy_stage > 0.99:
            # record templates for SCAN
            if dataset == 'SCAN':
                if curriculum_stage == 2:
                    for i in [12, 13, 14, 15]:
                        model.encoder.treelstm_layer.record_template_scan(i, 2)
                if curriculum_stage == 3:
                    for i in [9, 10, 16, 17, 18, 19]:
                        model.encoder.treelstm_layer.record_template_scan(i, 3)
            # record templates for COGS
            if dataset == "COGS":
                model.encoder.treelstm_layer.record_template_cogs()
            curriculum_stage += 3
            if dataset == "SCAN":
                filter_fn = lambda x: len(x[0]) == curriculum_stage
            elif dataset == "COGS":
                filter_fn = lambda x: curriculum_stage-3 < len(x[0]) <= curriculum_stage
            train_data_curriculum = list(filter(filter_fn, train_data))
            print(train_data_curriculum)
            print("curriculum size: ", len(train_data_curriculum))
            preprocessed_train_data = preprocess(train_data_curriculum, x_vocab, y_vocab)
            train_loader = MyDataLoader(preprocessed_train_data,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                                        y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                                        max_x_seq_len=max_x_seq_len,
                                        max_y_seq_len=max_y_seq_len)

            test_data_curriculum = list(filter(filter_fn, test_data))
            print(test_data_curriculum)
            preprocessed_valid_data = preprocess(test_data_curriculum, x_vocab, y_vocab)
            valid_loader = MyDataLoader(preprocessed_valid_data,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                                        y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                                        max_x_seq_len=max_x_seq_len,
                                        max_y_seq_len=max_y_seq_len)
            train_accuracy_stage_list = []
            train_accuracy_stage = 0.0
            iter_count_stage = 0
            validated = False
            # TK worst_train_accuracy_epoch = 0.0
        for batch_iter, train_batch in enumerate(train_loader):
            if args.use_curriculum:
                print("\nstage: ", curriculum_stage)
            print("\niteration:", iter_count)
            train_loss, train_accuracy = run_iter(batch=train_batch,
                                                    average_accuracy=train_accuracy_stage,
                                                    is_training=True,
                                                    verbose=True)
            train_accuracy_stage_list.append(train_accuracy)
            if len(train_accuracy_stage_list) == 501:
                train_accuracy_stage_list = train_accuracy_stage_list[1:]
            train_accuracy_stage_total = sum(train_accuracy_stage_list)
            train_accuracy_stage = train_accuracy_stage_total / len(train_accuracy_stage_list)

            print("iteration loss:           %1.4f" %train_loss.item())
            print("iteration train accuracy: %1.4f" %train_accuracy.item())
            print("stage train accuracy:     %1.3f" %train_accuracy_stage)
            # print("worst batch training accuracy for the epoch: %1.4f" % worst_train_accuracy_epoch)
            add_scalar_summary(summary_writer=train_summary_writer, name='loss', value=train_loss, step=iter_count)
            add_scalar_summary(summary_writer=train_summary_writer, name='accuracy', value=train_accuracy, step=iter_count)

            iter_count += 1
            iter_count_stage += 1
            if (iter_count + 1) % 10 == 0:
                #temp = max(1.0 - train_accuracy_stage, 0.5)
                temp = max(10.0 - train_accuracy_stage * 10, 0.5)
                model.reset_gumbel_temp(temp)

            # validate once for each stage
            if (not validated and train_accuracy_stage > 0.95 and curriculum_stage != 1):
                validated = True
                valid_loss_sum = valid_accuracy_sum = 0
                num_valid_batches = valid_loader.num_batches
                k = randint(0, num_valid_batches - 1)
                for i, valid_batch in enumerate(valid_loader):
                    if args.print_in_valid and i == k:
                        valid_loss, valid_accuracy = run_iter(batch=valid_batch,
                                                              average_accuracy=train_accuracy_stage,
                                                              verbose=True)
                    else:
                        valid_loss, valid_accuracy = run_iter(batch=valid_batch,
                                                              average_accuracy=train_accuracy_stage,
                                                              verbose=True)
                    valid_loss_sum += valid_loss.item()
                    valid_accuracy_sum += valid_accuracy.item()
                valid_loss = valid_loss_sum / num_valid_batches
                valid_accuracy = valid_accuracy_sum / num_valid_batches
                add_scalar_summary(summary_writer=valid_summary_writer, name='loss', value=valid_loss, step=iter_count)
                add_scalar_summary(summary_writer=valid_summary_writer, name='accuracy', value=valid_accuracy,
                                   step=iter_count)
                scheduler.step(valid_accuracy)
                progress = iter_count / train_loader.num_batches
                logging.info(
                    f'Epoch {progress:.2f}: valid loss = {valid_loss:.4f}, valid accuracy = {valid_accuracy:.4f}')
                # TK DEBUG
                print("semantic loss: %1.4f" % valid_loss)
                print("train accuracy: %1.4f" % valid_accuracy)
                if valid_accuracy >= best_vaild_accuacy:
                    best_vaild_accuacy = valid_accuracy
                    model_filename = (f'model-{progress:.2f}'
                                      f'-{valid_loss:.4f}'
                                      f'-{valid_accuracy:.4f}.pkl')
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    print(f'Saved the new best model to {model_path}')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
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
    parser.add_argument('--dev-set', type=str)
    parser.add_argument('--test-set', type=str)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
