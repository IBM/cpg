from tqdm import tqdm

from src.model.cogs_data import COGSDataset


def create():
    dataset = COGSDataset(30, 30)
    cogs_few_shot_file = open('./cogs_data/cogs_train_few_shot.tsv', 'w')
    
    saved_types = []

    for input_length in tqdm(range(2, 20)):
        with open("cogs_data/cogs_train.tsv", "rt") as file:
            for line in file:
                line_parsed = line.split('\t')
                input = line_parsed[:2][0].replace(".", "").lower()
                if len(input.split(' ')) != input_length:
                    continue
                _, types, _ = dataset.parse(input)

                if any((t not in saved_types for t in types)):
                    cogs_few_shot_file.write(line)
                
                saved_types.extend(types)
    
    cogs_few_shot_file.close()


def main():
    create()


if __name__ == '__main__':
    main()