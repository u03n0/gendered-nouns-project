import torch

from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd

from models import NounDataset, GenderBert, GenderedCNN
from model_setup import CONFIGURATION
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split



def build_dataset_from_args(lang_list, df):
    if len(lang_list) == 1:
        return df[df['lang'] == lang_list[0]]
    return pd.concat([df[df['lang'] == lang] for lang in lang_list])


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def get_x_y_from(df):
    return df['noun'].tolist(), df['gender'].tolist()


def initialize_classifier(model_name, hyperparameters):
    MODEL_DICT = {
        'bert': GenderBert(**hyperparameters),
        # 'cnn': GenderedCNN(**hyperparameters)
    }
    return MODEL_DICT[model_name]


def verify_args_are_valid(args, df: pd.DataFrame)-> bool:
    """ 
    """
    all_args = [elem for arg_value in vars(args).values() for elem in arg_value]
    possibilities = possible_options(df)
    return all(lang in possibilities for lang in all_args)


def possible_options(df: pd.DataFrame)-> List[str]:
    languages = df['lang'].unique().tolist()
    models = list(CONFIGURATION.keys())
    return languages + models


def get_pretrained_file(args, model_):
    langs = "-".join([lang for lang in args])
    folder_path = '../saved_models'
    file_name = f"{model_}_trained_on_{langs}.pth"
    return Path(folder_path) / file_name


def load_pretrained_model(model_name, hyperparameters, path):
    clf = initialize_classifier(model_name, hyperparameters)
    checkpoint = torch.load(path)
    clf.model.load_state_dict(checkpoint['model_state_dict'])
    clf.tokenizer = checkpoint['tokenizer']
    print(f'Model loaded from {path}')
    return clf

def build_dataloaders(args, data, config):
    if args.train == args.evaluate:
        df_from_args = build_dataset_from_args(args.train, data)
        X, y = get_x_y_from(df_from_args)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        train_loader = x_y_to_dataloader(X_train, y_train, config)
        test_loader = x_y_to_dataloader(X_test, y_test, config)
        return train_loader, test_loader
    else:
        df_from_train_args = build_dataset_from_args(args.train, data)
        df_from_test_args = build_dataset_from_args(args.test, data)
        X_train, y_train = get_x_y_from(df_from_train_args)
        X_test, y_test = get_x_y_from(df_from_test_args)
        train_loader = x_y_to_dataloader(X_train, y_train, config)
        train_loader = x_y_to_dataloader(X_test, y_test, config)
    return train_loader, test_loader
    
def x_y_to_dataloader(X, y, config):
    """ helper function
    """
    ds = NounDataset(X, y, config['tokenizer'], config['max_length'])
    return DataLoader(ds, batch_size=config['batch_size'], shuffle=False)

def save_model(model, path,):
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': model.tokenizer,
        }, path)
        print(f'Model saved as {path}')