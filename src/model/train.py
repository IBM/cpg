import argparse
import os
import dill
from tqdm import tqdm
import wandb
import random
import numpy as np

import torch
from torch import optim
from torch.optim import lr_scheduler

from src.model.module import CompositionalLearner
from src.model.scan_data import SCANDataset
from src.model.cogs_data import COGSDataset
from src.model.basic import run_iter


def train(args):
    if args.seed != -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        print('Running with seed:', args.seed)

    if args.dataset == 'SCAN':
        dataset = SCANDataset(args.word_dim, args.template_dim)
    elif args.dataset == 'COGS':
        dataset = COGSDataset(args.word_dim, args.template_dim)

    train_data, valid_data, x_vocab, y_vocab = dataset.get_data(args.training_set, args.validation_set)

    wandb.init(
        project="CPG",
        
        # track hyperparameters and run metadata
        config={
        "training set size": len(train_data),
        "validation set size": len(valid_data),
        "x vocab size": len(x_vocab),
        "y vocab size": len(y_vocab),
        }
    )

    curriculum_stage = dataset.get_next_curriculum_stage()
    train_loader = dataset.load_data(train_data, curriculum_stage, args.batch_size)
    valid_loader = dataset.load_data(valid_data, curriculum_stage, args.batch_size)

    model = CompositionalLearner(dataset=dataset, gumbel_temp=10.0)
    model.to('cpu')

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

    epoch_stage = 0
    iteration_total = 0
    iteration_stage = 0
    train_accuracy_stage_list = []
    train_accuracy_stage = 0.0
    for _ in range(300000):
        epoch_stage += 1
        for train_batch in tqdm(train_loader, total=train_loader.num_batches):
            iteration_total += 1
            iteration_stage += 1
            model.reset_hyperparameters(train_accuracy_stage, iteration_stage)

            if args.verbose:
                print(f'\n\ncurriculum stage {curriculum_stage[0]} to {curriculum_stage[1]}')
                print("iterations in current stage:", iteration_stage)
                print("epochs in current stage:    ", epoch_stage)
                print("total iterations:           ", iteration_total)
            _, train_accuracy = run_iter(model, train_batch, params, optimizer, is_training=True, verbose=args.verbose)
            train_accuracy_stage_list.append(train_accuracy)
            if len(train_accuracy_stage_list) == 1001:
                train_accuracy_stage_list = train_accuracy_stage_list[1:]
            train_accuracy_stage = sum(train_accuracy_stage_list) / len(train_accuracy_stage_list)
            if args.verbose:
                print("iteration train accuracy: %1.4f" %train_accuracy.item())
                print("stage train accuracy:     %1.3f" %train_accuracy_stage)
            wandb.log({'iteration_total': iteration_total, 'iteration_train_accuracy':train_accuracy, 'stage_train_accuracy':train_accuracy_stage})

        if train_accuracy_stage == 1. or train_loader.num_batches == 0:
            model.start_eval()
            model.record_templates()

            # validate
            valid_accuracy_sum = 0
            num_valid_batches = valid_loader.num_batches
            iteration_valid = 0
            if num_valid_batches != 0:
                for valid_batch in tqdm(valid_loader, total=num_valid_batches):
                    iteration_valid += 1
                    _, valid_accuracy = run_iter(model, valid_batch, print_error=True)
                    valid_accuracy_sum += valid_accuracy.item()
                    wandb.log({'iteration_valid': iteration_valid, 'validation_accuracy':valid_accuracy})
                valid_accuracy = valid_accuracy_sum / num_valid_batches
                wandb.log({'iteration_valid': iteration_valid, 'average_validation-accuracy':valid_accuracy})
                scheduler.step(valid_accuracy)
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
    parser.add_argument('--max-epoch', default=300000, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--word-dim', default=30, type=int)
    parser.add_argument('--template-dim', default=30, type=int)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--halve-lr-every', default=2, type=int)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--training-set', required=True, type=str)
    parser.add_argument('--validation-set', required=True, type=str)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
