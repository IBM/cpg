import argparse
import logging
import os
import dill
from tensorboardX import SummaryWriter

import torch
from torch import optim
from torch.optim import lr_scheduler

from src.model.module import CompositionalLearner
from src.model.scan_data import SCANDataset
from src.model.cogs_data import COGSDataset
from src.model.basic import run_iter, add_scalar_summary


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')


def train(args):
    if args.dataset == 'SCAN':
        dataset = SCANDataset(args.word_dim, args.template_dim)
    elif args.dataset == 'COGS':
        dataset = COGSDataset(args.word_dim, args.template_dim)

    train_data, valid_data, x_vocab, y_vocab = dataset.get_data(args.training_set, args.validation_set)

    logging.info(f"Training set size: {len(train_data)}")
    logging.info(f"Training set sample:\n\tx: {train_data[0][0]}\n\ty: {train_data[0][1]}")
    logging.info(f"Validation set size: {len(valid_data)}")
    logging.info(f"Validation set sample:\n\tx: {valid_data[0][0]}\n\ty: {valid_data[0][1]}")
    logging.info(f"X Vocab size: {len(x_vocab)}")
    logging.info(f"Y Vocab size: {len(y_vocab)}")

    curriculum_stage = dataset.get_next_curriculum_stage()
    train_loader = dataset.load_data(train_data, curriculum_stage, args.batch_size)
    valid_loader = dataset.load_data(valid_data, curriculum_stage, args.batch_size)

    model = CompositionalLearner(dataset=dataset, gumbel_temp=10.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    optimizer = optimizer_class(params=params)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5,
                                               patience=20 * args.halve_lr_every, verbose=True)

    train_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'train'))
    valid_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'valid'))

    epoch_stage = 0
    iteration_total = 0
    iteration_stage = 0
    train_accuracy_stage_list = []
    train_accuracy_stage = 0.0
    for _ in range(300000):
        for train_batch in train_loader:
            model.reset_hyperparameters(train_accuracy_stage, iteration_stage)

            print(f'\n\ncurriculum stage {curriculum_stage[0]} to {curriculum_stage[1]}')
            print("iterations in current stage:", iteration_stage)
            print("epochs in current stage:    ", epoch_stage)
            print("total iterations:           ", iteration_total)
            _, train_accuracy = run_iter(model, train_batch, params, optimizer, is_training=True, verbose=True)
            train_accuracy_stage_list.append(train_accuracy)
            if len(train_accuracy_stage_list) == 501:
                train_accuracy_stage_list = train_accuracy_stage_list[1:]
            train_accuracy_stage = sum(train_accuracy_stage_list) / len(train_accuracy_stage_list)
            print("iteration train accuracy: %1.4f" %train_accuracy.item())
            print("stage train accuracy:     %1.3f" %train_accuracy_stage)
            add_scalar_summary(summary_writer=train_summary_writer, name='accuracy', value=train_accuracy, step=iteration_total)

            iteration_total += 1
            iteration_stage += 1

        epoch_stage += 1

        if train_accuracy_stage > 0.99:
            model.start_eval()
            model.record_templates()

            # validate
            valid_accuracy_sum = 0
            num_valid_batches = valid_loader.num_batches
            if num_valid_batches != 0:
                for valid_batch in valid_loader:
                    _, valid_accuracy = run_iter(model, valid_batch, verbose=True)
                    valid_accuracy_sum += valid_accuracy.item()
                valid_accuracy = valid_accuracy_sum / num_valid_batches
                add_scalar_summary(summary_writer=valid_summary_writer, name='accuracy', value=valid_accuracy, step=iteration_total)
                scheduler.step(valid_accuracy)
                logging.info(
                    f'curriculum stage {curriculum_stage[0]} to {curriculum_stage[1]}, epoch {epoch_stage:.2f}: valid accuracy = {valid_accuracy:.4f}')
                model_filename = (f'model-{curriculum_stage[0]}-{curriculum_stage[1]}-{epoch_stage:.2f}-{valid_accuracy:.4f}.pkl')
                model_path = os.path.join(args.save_dir, model_filename)
                torch.save(model, model_path, pickle_module=dill)
                print(f'Saved the latest model to {model_path}')
            
            # pass to the next curriculum stage
            curriculum_stage = dataset.get_next_curriculum_stage()
            if curriculum_stage == None:
                break
            train_loader = dataset.load_data(train_data, curriculum_stage, args.batch_size)
            valid_loader = dataset.load_data(valid_data, curriculum_stage, args.batch_size)

            epoch_stage = 0
            iteration_stage = 0
            train_accuracy_stage_list = []
            train_accuracy_stage = 0.0

def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--max-epoch', required=True, type=int)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--template-dim', required=True, type=int)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--halve-lr-every', default=2, type=int)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--print-in-valid', default=False, action='store_true')
    parser.add_argument('--use-curriculum', default=False, action='store_true')
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--training-set', required=True, type=str)
    parser.add_argument('--validation-set', required=True, type=str)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
