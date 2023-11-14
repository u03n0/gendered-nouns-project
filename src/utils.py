import torch

from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd

from models import NounDataset, GenderBert, GenderedCNN
from model_setup import CONFIGURATION



def build_dataset(lang_list, df):
    if len(lang_list) == 1:
        return df[df['lang'] == lang_list[0]]
    return pd.concat([df[df['lang'] == lang] for lang in lang_list])


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def x_y(df):
    return df['noun'].tolist(), df['gender'].tolist()


def compile_dataset(X, y, tokenizer, max_length):
    return NounDataset(X, y, tokenizer, max_length)


def get_classifier(model_name, hyperparameters):
    MODEL_DICT = {
        'bert': GenderBert(**hyperparameters),
        # 'cnn': GenderedCNN(**hyperparameters)
    }
    return MODEL_DICT[model_name]


def verify_args(args, df: pd.DataFrame)-> bool:
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
    src_langs = "-".join([lang for lang in args.train])
    tgt_langs = "-".join([lang for lang in args.evaluate])
    folder_path = '../saved_models'
    file_name = f"{model_}_{src_langs}_to_{tgt_langs}.pth"
    return Path(folder_path) / file_name


def load_pretrained_model(model_name, hyperparameters, file_path):
    clf = get_classifier(model_name, hyperparameters)
    clf.load_state_dict(torch.load(file_path), strict=False)
    print(f"Model loaded successfully from {file_path}.")
    return clf
