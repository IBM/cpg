import argparse
from tqdm import tqdm

import torch

from src.model.basic import run_iter


def evaluate(args):
    model = torch.load(args.model_path, map_location='cpu')
    dataset = model.dataset
    model.start_eval()
    model.to('cpu')
    torch.set_grad_enabled(False)

    _, test_data, x_vocab, y_vocab = dataset.get_data(args.training_set, args.test_set)

    if args.gen_eval:
        dataset.curriculum = [(i+2, i+2) for i in range(58)]
    else:
        dataset.reset_curriculum()

    curriculum_stage = dataset.get_next_curriculum_stage()

    accuracy_sum_total = 0 # sum of accuracy for all test batches
    num_batches_total = 0 # number of all test batches
    while curriculum_stage != None:
        if curriculum_stage[0] >= 16:
            dataset.max_x_seq_len = 60
            dataset.max_y_seq_len = 500
        test_loader = dataset.load_data(test_data, curriculum_stage, args.batch_size)
        test_accuracy_sum = 0
        num_test_batches = test_loader.num_batches
        if num_test_batches != 0:
            for test_batch in tqdm(test_loader, total=num_test_batches):
                _, test_accuracy = run_iter(model, test_batch, verbose=args.verbose, print_error=True)
                test_accuracy_sum += test_accuracy.item()
            test_accuracy = test_accuracy_sum / num_test_batches
            accuracy_sum_total += test_accuracy_sum
            num_batches_total += num_test_batches
            test_accuracy_total = accuracy_sum_total / num_batches_total
            print(f'curriculum stage {curriculum_stage[0]} to {curriculum_stage[1]}: '
                         f'test accuracy = {test_accuracy:.4f}, total test accuracy = {test_accuracy_total:.4f}')
        
        # pass to the next curriculum stage
        curriculum_stage = dataset.get_next_curriculum_stage()


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--max-epoch', default=300000, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--training-set', required=True, type=str)
    parser.add_argument('--test-set', required=True, type=str)
    parser.add_argument('--gen-eval', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
