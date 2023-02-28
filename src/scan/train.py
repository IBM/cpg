import argparse
import logging
import os
from random import randint

from tensorboardX import SummaryWriter

import torch
import numpy as np
from torch import nn, optim
from torch.optim import lr_scheduler, optimizer
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F

from src.scan.model import SCANModel

from src.scan.data import load_SCAN_length, load_SCAN_simple, build_vocab, preprocess, MyDataLoader, \
                          load_SCAN_add_jump_0, load_SCAN_add_jump_4, parse_scan, \
        load_SCAN_add_jump_0_no_jump_oversampling

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def train(args):
    # load train and test data
    train_data, test_data = load_SCAN_add_jump_0_no_jump_oversampling()
    training_size = int(len(train_data) * args.data_frac)
    train_data = train_data[:training_size]
    logging.info(f"Train data set size: {len(train_data)}")
    logging.info(f"Train data sample:\n\tx: {train_data[0][0]}\n\ty: {train_data[0][1]}")
    logging.info(f"Test data set size: {len(test_data)}")
    logging.info(f"Test data sample:\n\tx: {test_data[0][0]}\n\ty: {test_data[0][1]}")
    x_vocab = build_vocab([x for x, _ in train_data],
                           base_tokens=['<PAD>', '<UNK>'])
    y_vocab = build_vocab([y for _, y in train_data],
                           base_tokens=['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
    logging.info(f"X Vocab size: {len(x_vocab)}")
    logging.info(f"Y Vocab size: {len(y_vocab)}")

    def select(x, curriculum_stage):
        if curriculum_stage < 4:
            if "and" in x[0] or "after" in x[0]:
                return False
        if curriculum_stage < 3:
            if "twice" in x[0] or "thrice" in x[0]:
                return False
        if curriculum_stage < 2:
            if "opposite" in x[0] or "around" in x[0]:
                return False
        if curriculum_stage < 1:
            if "turn" in x[0]:
                return False
        return True

    if args.use_curriculum:
        # TK DEBUG -- changed from 0
        curriculum_stage = 1
        # filter_fn = lambda x: select(x, curriculum_stage)
        filter_fn = lambda x: len(x[0]) <= curriculum_stage
        train_data_curriculum = list(filter(filter_fn, train_data))
        print(train_data_curriculum)
        preprocessed_train_data = preprocess(train_data_curriculum, x_vocab, y_vocab)
        valid_data_curriculum = list(filter(filter_fn, test_data))
        print(valid_data_curriculum)
        preprocessed_valid_data = preprocess(valid_data_curriculum, x_vocab, y_vocab)

    else:
        preprocessed_train_data = preprocess(train_data, x_vocab, y_vocab)
        preprocessed_valid_data = preprocess(test_data, x_vocab, y_vocab)
    train_loader = MyDataLoader(preprocessed_train_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                                y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                                max_x_seq_len=args.max_x_seq_len,
                                max_y_seq_len=args.max_y_seq_len)

    valid_loader = MyDataLoader(preprocessed_valid_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                                y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                                max_x_seq_len=args.max_x_seq_len,
                                max_y_seq_len=args.max_y_seq_len)
    
    model = SCANModel(model=args.model,
                      y_vocab=y_vocab,
                      x_vocab=x_vocab,
                      word_dim=args.word_dim,
                      hidden_value_dim=args.hidden_dim,
                      hidden_type_dim=26,
                      decoder_hidden_dim=args.decoder_hidden_dim,
                      decoder_num_layers=args.decoder_num_layers,
                      use_leaf_rnn=args.leaf_rnn,
                      bidirectional=args.bidirectional,
                      intra_attention=args.intra_attention,
                      use_batchnorm=args.batchnorm,
                      dropout_prob=args.dropout,
                      max_y_seq_len=args.max_y_seq_len,
                      use_prim_type_oracle=args.use_prim_type_oracle,
                      syntactic_supervision=args.syntactic_supervision)
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

    def run_iter(batch, average_accuracy, is_training=False, use_hom_loss=False, verbose=False):
        model.train(is_training)
        batch_x, batch_y = batch
        B, L = batch_x.size()
        if args.syntactic_supervision:
            positions_force = []
            types_force = []
            x_tokens = x_vocab.decode_batch(batch_x.numpy(), batch_x != x_vocab.token_to_idx('<PAD>'))
            for i in range(len(x_tokens)):
                x_tokens[i] = " ".join(x_tokens[i])
            #print("x tokens:", x_tokens)
            for i in range(B):
                positions, types = parse_scan(x_tokens[i])
                positions_force.append(positions)
                types_force.append(types)
            #print("types force:", types_force)
        else:
            positions_force = None
            types_force = None
        if is_training and args.use_teacher_forcing:
            force = batch_y
        else:
            force = None
        decoding, hom_loss, dt_all, logits_init = model(x=batch_x, length=torch.full((B, 1), L).view(B),
                                                        force=force, positions_force=positions_force, types_force=types_force)
        decoding_idx = decoding.argmax(-1)
        if verbose:
            input = x_vocab.decode_batch(batch_x.numpy(), batch_x != x_vocab.token_to_idx('<PAD>'))
            expected = y_vocab.decode_batch(batch_y.numpy(), batch_y != y_vocab.token_to_idx('<PAD>'))
            decoded = y_vocab.decode_batch(decoding_idx.numpy(), decoding_idx != y_vocab.token_to_idx('<PAD>'))
            print("--------------------------------")
            for i in range(B):
                print("input: ", ", ".join(input[i]), "\n")
                print("expected: ", ", ".join(expected[i]), "\n")
                print("decoded: ", ", ".join(decoded[i]))
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
        success = torch.eq(match.sum(1), match.size(1)).float()
        match = [(match[i].sum() == match.size(1)).float() for i in range(match.size(0))]

        # compute cross entropy loss of the decodings against the ground truth

        loss = F.cross_entropy(torch.flatten(decoding, start_dim=0, end_dim=1),
                               torch.flatten(expected_padded.long(), start_dim=0, end_dim=1), ignore_index=0)
        print("cross entropy loss: ", loss)
        # compute template loss
        # dt-all: B x ? x 8 (probs -- verify)
        # match: B x max_len (49)
        # success: B
        # probs, idx: B x ?
        # use multinomial distribution
        #B, T, K, _ = dt_all.size()
        #dt_all_idx = torch.multinomial(torch.softmax(dt_all, -1).view(B * T * K, 3), 1).view(B, T, K)
        #probs = torch.gather(dt_all, 3, dt_all_idx.unsqueeze(3)).squeeze()
        temp = max(1.0-average_accuracy, 0.5)
        dt_all = torch.nn.functional.gumbel_softmax(dt_all, tau=temp, hard=False)
        probs, _ = torch.max(dt_all, dim=-1)
        log_probs = torch.clamp(probs, min=1.0e-20, max=1.0).log()
        B, T, temp_vocab_size = probs.size()
        success_per_decoding = success.unsqueeze(1).unsqueeze(2).expand(B, T, temp_vocab_size)
        template_loss = ((success_per_decoding * -log_probs) + (1-success_per_decoding) * log_probs).mean()
        accuracy = torch.tensor(match).mean()
        # compute loss
        if is_training:
            optimizer.zero_grad()
            loss = loss + template_loss
            if use_hom_loss:
                loss += hom_loss
            loss.backward()
            clip_grad_norm_(parameters=params, max_norm=5)
            optimizer.step()
        return loss, accuracy, hom_loss

    def add_scalar_summary(summary_writer, name, value, step):
        if torch.is_tensor(value):
            value = value.item()
        summary_writer.add_scalar(tag=name, scalar_value=value, global_step=step)

    # num_train_batches = train_loader.num_batches
    # validate_every = num_train_batches * 3
    best_vaild_accuacy = 0
    iter_count = 1
    train_accuracy_epoch_total = 0.0
    train_accuracy_epoch = 0.0
    for e in range(args.max_epoch):
        if args.use_curriculum and train_accuracy_epoch > .9:
            curriculum_stage += 1
            model.reset_gumbel_temp(1.0)
            #filter = lambda x: select(x, curriculum_stage
            filter_fn = lambda x: len(x[0]) <= curriculum_stage
            train_data_curriculum = list(filter(filter_fn, train_data))
            print(train_data_curriculum)
            preprocessed_train_data = preprocess(train_data_curriculum, x_vocab, y_vocab)
            train_loader = MyDataLoader(preprocessed_train_data,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                                        y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                                        max_x_seq_len=args.max_x_seq_len,
                                        max_y_seq_len=args.max_y_seq_len)

            test_data_curriculum = list(filter(filter_fn, test_data))
            print(test_data_curriculum)
            preprocessed_valid_data = preprocess(test_data_curriculum, x_vocab, y_vocab)
            valid_loader = MyDataLoader(preprocessed_valid_data,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                                        y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                                        max_x_seq_len=args.max_x_seq_len,
                                        max_y_seq_len=args.max_y_seq_len)
            train_accuracy_epoch_total = 0.0
            # TK worst_train_accuracy_epoch = 0.0
        for batch_iter, train_batch in enumerate(train_loader):
            if args.use_curriculum:
                print("\nstage: ", curriculum_stage)
            print("\niteration:", iter_count)
            if iter_count > -1:
                train_loss, train_accuracy, hom_loss = run_iter(batch=train_batch,
                                                                average_accuracy=train_accuracy_epoch,
                                                                is_training=True,
                                                                use_hom_loss=True,
                                                                verbose=True)
                print("homomorphic loss: %1.4f" %hom_loss.item())
            else:
                train_loss, train_accuracy, _ = run_iter(batch=train_batch,
                                                         average_accuracy=train_accuracy_epoch,
                                                         is_training=True)
            train_accuracy_epoch_total += train_accuracy
            train_accuracy_epoch = train_accuracy_epoch_total / (iter_count+1)
            # if train_accuracy_epoch < worst_train_accuracy_epoch:
            #     worst_train_accuracy_epoch = train_accuracy_epoch

            print("iteration loss: %1.4f" %train_loss.item())
            print("iteration train accuracy: %1.4f" %train_accuracy.item())
            print("average train accuracy for the epoch: %1.4f" % train_accuracy_epoch)
            # print("worst batch training accuracy for the epoch: %1.4f" % worst_train_accuracy_epoch)
            add_scalar_summary(summary_writer=train_summary_writer, name='loss', value=train_loss, step=iter_count)
            add_scalar_summary(summary_writer=train_summary_writer, name='accuracy', value=train_accuracy, step=iter_count)

            iter_count += 1
            if (iter_count + 1) % 500 == 0:
                model.reduce_gumbel_temp(iter_count)
            # TK DEBUG
            if iter_count == 1000:
                pass
            # validate every epoch at the start
            if e % 3 == 0 and batch_iter == 1 and train_accuracy_epoch > 0.89:
                valid_loss_sum = valid_accuracy_sum = 0
                num_valid_batches = valid_loader.num_batches
                k = randint(0, num_valid_batches - 1)
                for i, valid_batch in enumerate(valid_loader):
                    if args.print_in_valid and i == k:
                        valid_loss, valid_accuracy, _ = run_iter(batch=valid_batch,
                                                                 average_accuracy=train_accuracy_epoch,
                                                                 verbose=True)
                    else:
                        valid_loss, valid_accuracy, _ = run_iter(batch=valid_batch,
                                                                 average_accuracy=train_accuracy_epoch,
                                                                 verbose=True)
                    valid_loss_sum += valid_loss.item()
                    valid_accuracy_sum += valid_accuracy.item()
                valid_loss = valid_loss_sum / num_valid_batches
                valid_accuracy = valid_accuracy_sum / num_valid_batches
                add_scalar_summary(summary_writer=valid_summary_writer, name='loss', value=valid_loss, step=iter_count)
                add_scalar_summary(summary_writer=valid_summary_writer, name='accuracy', value=valid_accuracy, step=iter_count)
                scheduler.step(valid_accuracy)
                progress = iter_count / train_loader.num_batches
                logging.info(f'Epoch {progress:.2f}: valid loss = {valid_loss:.4f}, valid accuracy = {valid_accuracy:.4f}')
                # TK DEBUG
                print("semantic loss: %1.4f" % valid_loss)
                print("train accuracy: %1.4f" % valid_accuracy)
                if valid_accuracy > best_vaild_accuacy:
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
    parser.add_argument('--max_x_seq_len', default = 9)
    parser.add_argument('--max_y_seq_len', default = 49)
    parser.add_argument('--data-frac', default = 1., type=float)
    parser.add_argument('--use_teacher_forcing', default=True)
    parser.add_argument('--model', default='tree-lstm', choices={'tree-lstm', 'lstm'})
    parser.add_argument('--use-prim-type-oracle', default=False)
    parser.add_argument('--print-in-valid', default=False)
    parser.add_argument('--use-curriculum', default=False)
    parser.add_argument('--syntactic-supervision', default=False)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
