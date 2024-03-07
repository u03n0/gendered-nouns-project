import torch
import argparse

import pandas as pd

from sklearn.model_selection import train_test_split

from utils import (verify_args_are_valid, build_lang_df, get_pretrained_file, possible_options,
                   process_data, save_metadata, get_x_y_from, build_dataloader)
from models import Bert

if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

    parser = argparse.ArgumentParser() # Argument Parser
    parser.add_argument('-t', '--train', help='Specify which languages to train on : es, fr, de, etc', nargs='+', type=str, required=True, action='store')
    parser.add_argument('-e', '--evaluate', help='Specify which languages to evaluate on : es, fr, de, etc', nargs='+', type=str, required=True, action='store')
    parser.add_argument('-m', '--model', help='Specify which model to use : cnn, bert, etc', nargs='+', type=str, required=True, action='store')

    args = parser.parse_args()

    df = pd.read_csv("data/wiktionary_raw.csv")
    valid_args = verify_args_are_valid(args, df)

    if valid_args:
        data = process_data(df)
        for model_ in args.model:
            df_train_lang = build_lang_df(args.train, data)
            df_test_lang = build_lang_df(args.evaluate, data)
            num_labels = len(df_train_lang['gender'].unique())
            pretrained_model_path = get_pretrained_file(args.train, model_)
            model_class = globals()[model_]
            clf = model_class(num_labels)

            if pretrained_model_path.exists():
                clf.load_model(pretrained_model_path)
                X, y = get_x_y_from(df_test_lang)
                test_loader = build_dataloader(X, y, 32, 34, clf)
            else:
                print(f"{model_} model will be trained on {args.train} which has {num_labels} genders, using {device}")
                if args.train == args.evaluate:
                    X, y = get_x_y_from(df_train_lang)
                    X_train, X_test, y_train, y_test = train_test_split(X, y) 
                else:
                    X_train, y_train = get_x_y_from(df_train_lang)
                    X_test, y_test = get_x_y_from(df_test_lang)
                train_loader = build_dataloader(X_train, y_train, 32, 34, clf)
                test_loader = build_dataloader(X_test, y_test, 32, 34, clf)

                clf.train_model(train_loader, device=device, num_epochs=8)
                clf.save_model(pretrained_model_path)
            
            print(f"Testing {model_} model on {args.evaluate} using {device}")
            results = clf.evaluate(test_loader, device=device)
            save_metadata(results, model_, args)
    else:
        print(f"Invalid arguments: please select from {possible_options(df)}")