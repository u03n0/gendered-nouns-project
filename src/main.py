import torch
import argparse

import pandas as pd

from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split

import utils
from hyperparameters_dev import HYPERPARAMETERS
from model_setup import CONFIGURATION


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', help='Specify which languages to train on : es, fr, de, etc', nargs='+', type=str, required=True, action='store')
    parser.add_argument('-e', '--evaluate', help='Specify which languages to evaluate on : es, fr, de, etc', nargs='+', type=str, required=True, action='store')
    parser.add_argument('-m', '--model', help='Specify which model to use : cnn, bert, etc', nargs='+', type=str, required=True, action='store')

    args = parser.parse_args()

    # Read the large raw data csv file as a DataFrame
    df = pd.read_csv("../data/wiktionary_raw.csv")

    # Verify that arguments are valid: languages selected exist in dataset, model exists, etc
    valid_args = utils.verify_args(args, df)
    if valid_args:
        # preprocess raw data
        data = utils.clean_data(df)
        # build train set from language(s) specified
        train = utils.build_dataset(args.train, data)
        # build test set from language(s) specified
        test = utils.build_dataset(args.evaluate, data)

        # get model(s) to use
        models = args.model
        for model_ in models:
            X, y = utils.x_y(train)
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            # setup training environment
            config = CONFIGURATION[model_]
            hyperparams = HYPERPARAMETERS[model_]

            # Check if model with specific languages is already trained
            pretrained_model_path = utils.get_pretrained_file(args, model_)
            if pretrained_model_path.exists():
                if config['uses_hf']: # could be removed if all models use huggingface
                    clf = utils.load_pretrained_model(model_, hyperparams, pretrained_model_path)
                    # hf_model = config['model'].from_pretrained(config['model_name'], num_labels=len(set(y_train)))
                else:
                    # load the pretrained custom Model
                    pass
            else: # need to train the model since no pretrained data was found
                print(f"{model_} model will be trained on {args.train}")
                # Configure the setup for training model
                hf_model = config['model'].from_pretrained(config['model_name'], num_labels=len(set(y_train)))
                train_dataset = utils.compile_dataset(X_train, y_train, config['tokenizer'], config['max_length'])
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
                clf = utils.get_classifier(model_, hyperparams)
                # # train model(s)
                # clf.train_model(train_loader, device, config['epochs']) 
                # torch.save(hf_model.state_dict(), pretrained_model_path)
                # print(f"Successfully saved {model_} model")

                clf.train_model(train_loader, device=device, num_epochs=5)

            # creat test dataset and loader                
            test_dataset = utils.compile_dataset(X_test, y_test, config['tokenizer'], config['max_length'])
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
            # test model(s)
            print(f"Testing {model_} model on {args.evaluate}")
            results = clf.evaluate(test_loader, device=device)
    else:
        print(f"Invalid arguments: please select from {utils.possible_options(df)}")