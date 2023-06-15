import argparse
import logging
import os
from tensorboardX import SummaryWriter

import torch

from src.model.basic import run_iter, add_scalar_summary


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def evaluate(args):
    model = torch.load(args.model_path, map_location='cpu')
    dataset = model.dataset
    model.start_eval()
    model.to('cpu')
    torch.set_grad_enabled(False)

    _, test_data, x_vocab, y_vocab = dataset.get_data(args.training_set, args.test_set)

    dataset.reset_curriculum()
    #if evaluating on COGS generalization set
    #dataset.curriculum = [(i+2, i+2) for i in range(58)]
    #dataset.max_x_seq_len = 60
    #dataset.max_y_seq_len = 500

    logging.info(f"Test set size: {len(test_data)}")
    logging.info(f"Test set sample:\n\tx: {test_data[0][0]}\n\ty: {test_data[0][1]}")
    logging.info(f"X Vocab size: {len(x_vocab)}")
    logging.info(f"Y Vocab size: {len(y_vocab)}")

    curriculum_stage = dataset.get_next_curriculum_stage()

    test_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'valid'))
    
    accuracy_sum_total = 0 # sum of accuracy for all test batches
    num_batches_total = 0 # number of all test batches
    while curriculum_stage != None:
        test_loader = dataset.load_data(test_data, curriculum_stage, args.batch_size)
        test_accuracy_sum = 0
        num_test_batches = test_loader.num_batches
        if num_test_batches != 0:
            for test_batch in test_loader:
                _, test_accuracy = run_iter(model, test_batch, print_error=True)
                test_accuracy_sum += test_accuracy.item()
            test_accuracy = test_accuracy_sum / num_test_batches
            accuracy_sum_total += test_accuracy_sum
            num_batches_total += num_test_batches
            test_accuracy_total = accuracy_sum_total / num_batches_total
            add_scalar_summary(summary_writer=test_summary_writer, name='accuracy', value=test_accuracy, step=num_batches_total)
            logging.info(f'curriculum stage {curriculum_stage[0]} to {curriculum_stage[1]}: '
                         f'test accuracy = {test_accuracy:.4f}, total test accuracy = {test_accuracy_total:.4f}')
        
        # pass to the next curriculum stage
        curriculum_stage = dataset.get_next_curriculum_stage()


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--max-epoch', required=True, type=int)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--print-in-valid', default=False, action='store_true')
    parser.add_argument('--use-curriculum', default=False, action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--training-set', type=str)
    parser.add_argument('--test-set', type=str)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
