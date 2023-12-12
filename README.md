## For more details see the paper: Compositional Program Generation for Systematic Generalization
https://arxiv.org/abs/2309.16467

## Installation
```
pip install -r requirements.txt
```

## Training

### To train on SCAN
```
python -m src.model.train --save-dir pretrained/scan --dataset SCAN --training-set scan_data/SCAN_add_jump_0_train_no_jump_oversampling.txt --validation-set scan_data/SCAN_add_jump_4_test.txt
```

### To train on COGS
```
python -m src.model.train --save-dir pretrained/cogs --dataset COGS --training-set cogs_data/cogs_train_few_shot.tsv --validation-set cogs_data/cogs_dev.tsv
```

### Optional arguments
``--seed 1`` to set the random seed to 1

``--verbose`` to get detailed output and evaluation

``--wandb`` to use wandb

## Evaluation

### To evaluate on SCAN
```
python -m src.model.evaluate --model-path pretrained/scan/model-7-7-1.00-1.0000.pkl --save-dir pretrained/scan --dataset SCAN --training-set scan_data/SCAN_add_jump_0_train_no_jump_oversampling.txt --test-set scan_data/SCAN_add_jump_4_test.txt
```

### To evaluate on COGS
```
python -m src.model.evaluate --model-path pretrained\cogs\model-7-9-872.00-1.0000.pkl --save-dir pretrained/cogs --dataset COGS --training-set cogs_data/cogs_train_few_shot.tsv --test-set cogs_data/cogs_gen.tsv
```
