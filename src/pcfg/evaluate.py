import argparse

import numpy as np
import torch
from torch import nn

from src.pcfg.model import PCFGModel

from src.pcfg.data import build_vocab, preprocess, MyDataLoader, load_PCFG_split


def evaluate(args):
    train_data, test_data = load_PCFG_split(args.data_split)
    x_vocab = build_vocab([x for x, _ in train_data],
                           base_tokens=['<PAD>', '<UNK>'])
    y_vocab = build_vocab([y for _, y in train_data],
                           base_tokens=['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
    preprocessed_test_data = preprocess(test_data, x_vocab, y_vocab)
    test_loader = MyDataLoader(preprocessed_test_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                                y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                                max_x_seq_len=args.max_x_seq_len,
                                max_y_seq_len=args.max_y_seq_len)
    
    model = PCFGModel(model=args.model,
                      y_vocab=y_vocab,
                      x_vocab=x_vocab,
                      word_dim=args.word_dim,
                      hidden_value_dim=args.hidden_dim,
                      hidden_type_dim=10,
                      decoder_hidden_dim=args.decoder_hidden_dim,
                      decoder_num_layers=args.decoder_num_layers,
                      use_leaf_rnn=args.leaf_rnn,
                      bidirectional=args.bidirectional,
                      intra_attention=args.intra_attention,
                      use_batchnorm=args.batchnorm,
                      dropout_prob=args.dropout,
                      max_y_seq_len=args.max_y_seq_len,
                      use_prim_type_oracle=True)

    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.embedding.weight.size())
    print(f'# of parameters: {num_params}')
    print(f'# of word embedding parameters: {num_embedding_params}')
    print(f'# of parameters (excluding word embeddings): '
          f'{num_params - num_embedding_params}')
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    model.to(args.device)
    torch.set_grad_enabled(False)
    criterion = nn.CrossEntropyLoss()
    
    def run_iter(batch, is_training=False, verbose=False):
        model.train(is_training)
        batch_x, batch_y = batch
        B, L = batch_x.size()
        outputs, logits, _ = model(x=batch_x, length=torch.full((B, 1), L).view(B))
        if verbose:
            input = x_vocab.decode_batch(batch_x.numpy(), batch_x != x_vocab.token_to_idx('<PAD>'))
            expected = y_vocab.decode_batch(batch_y.numpy(), batch_y != y_vocab.token_to_idx('<PAD>'))
            decoded = y_vocab.decode_batch(outputs.numpy(), outputs != y_vocab.token_to_idx('<PAD>'))
            for i in range(B):
                print("input: ", ", ".join(input[i]), "\n")
                print("expected: ", ", ".join(expected[i]), "\n")
                print("decoded: ", ", ".join(decoded[i]))
                print("--------------------------------")
        _, N, V = logits.size()
        _, M = batch_y.size()
        if N >= M:
            # pad expected
            expected_padded = torch.full((B, N), y_vocab.token_to_idx('<PAD>'))
            expected_padded[:, :M] = batch_y
            outputs_padded = outputs
        else:
            # pad outputs
            expected_padded = batch_y
            outputs_padded = torch.full((B, M), y_vocab.token_to_idx('<PAD>'))
            outputs_padded[:, :N] = outputs
        # measure accuracy
        match = torch.eq(expected_padded, outputs_padded).float()
        match = [(match[i].sum() == match.size(1)).float() for i in range(match.size(0))]
        accuracy = torch.tensor(match).mean()
        # compute loss
        loss = criterion(input=logits.reshape(B * N, V), target=expected_padded[:, :N].reshape(B * N))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-split', default='localism',
                        choices={'all', 'localism', 'productivity', 'substitutivity', 'systematicity',
                                 'overgeneralization-0.005', 'overgeneralization-0.001',
                                 'overgeneralization-0.0005', 'overgeneralization-0.0001'})
    parser.add_argument('--model-path', required=True)
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
    parser.add_argument('--model', default='tree-lstm', choices={'tree-lstm', 'lstm'})
    parser.add_argument('--print-in-valid', default=False)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
