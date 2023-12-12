## To train on SCAN
``python -m src.model.train --save-dir pretrained/scan --dataset SCAN --training-set scan_data/SCAN_add_jump_0_train_no_jump_oversampling.txt --validation-set scan_data/SCAN_add_jump_4_test.txt``

## To evaluate on SCAN
``python -m src.model.evaluate --model-path pretrained/scan/model-7-7-1.00-1.0000.pkl --save-dir pretrained/scan --dataset SCAN --training-set scan_data/SCAN_add_jump_0_train_no_jump_oversampling.txt --test-set scan_data/SCAN_add_jump_4_test.txt``

## To train on COGS
``python -m src.model.train --save-dir pretrained/cogs --dataset COGS --training-set cogs_data/cogs_train_few_shot.tsv --validation-set cogs_data/cogs_dev.tsv``

## To evaluate on COGS
``python -m src.model.evaluate --model-path pretrained\cogs\model-7-9-872.00-1.0000.pkl --save-dir pretrained/cogs --dataset COGS --training-set cogs_data/cogs_train_few_shot.tsv --test-set cogs_data/cogs_gen.tsv``

## optional args: (1) --verbose to get detailed output and evaluation, (2) --wandb to use wandb