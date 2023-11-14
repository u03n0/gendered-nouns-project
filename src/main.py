import torch
import argparse

import pandas as pd


import utils
from hyperparameters_dev import HYPERPARAMETERS
from model_setup import CONFIGURATION


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

    parser = argparse.ArgumentParser() # Argument Parser
    parser.add_argument('-t', '--train', help='Specify which languages to train on : es, fr, de, etc', nargs='+', type=str, required=True, action='store')
    parser.add_argument('-e', '--evaluate', help='Specify which languages to evaluate on : es, fr, de, etc', nargs='+', type=str, required=True, action='store')
    parser.add_argument('-m', '--model', help='Specify which model to use : cnn, bert, etc', nargs='+', type=str, required=True, action='store')

    args = parser.parse_args()

    df = pd.read_csv("../data/wiktionary_raw.csv")
    valid_args = utils.verify_args_are_valid(args, df)

    if valid_args:
        data = utils.clean_data(df)
        for model_ in args.model:
            config = CONFIGURATION[model_] 
            hyperparams = HYPERPARAMETERS[model_]
            train_loader, test_loader, num_labels = utils.build_dataloaders(args, data, config)
            pretrained_model_path = utils.get_pretrained_file(args.train, model_)

            if pretrained_model_path.exists():
                clf = utils.initialize_classifier(model_, num_labels)
                clf.load_model(pretrained_model_path)
            else:
                print(f"{model_} model will be trained on {args.train} which has {num_labels} genders, using {device}")
                clf = utils.initialize_classifier(model_, num_labels)
                clf.train_model(train_loader, device=device, num_epochs=hyperparams['epochs'])
                clf.save_model(pretrained_model_path)

            print(f"Testing {model_} model on {args.evaluate} using {device}")
            results = clf.evaluate(test_loader, device=device)
            utils.save_metadata(results, model_, args)
    else:
        print(f"Invalid arguments: please select from {utils.possible_options(df)}")