import torch
import time
import json
import argparse
import inspect
from typing import Dict, List, Tuple, Type
from pathlib import Path

import pandas as pd

import models
from torch.utils.data import  DataLoader



def build_lang_df(lang_list: List, df: pd.DataFrame)-> pd.DataFrame:
    """ Creates a DataFrame by concatinating sub DataFrames based
    on each language provided.
    """
    if len(lang_list) == 1:
        return df[df['lang'] == lang_list[0]]
    return pd.concat([df[df['lang'] == lang] for lang in lang_list])


def process_data(df: pd.DataFrame)-> pd.DataFrame:
    """ Preprocesses a raw DataFrame by:
    - removing duplicates
    - dropping N/A
    - reducing all genders within each language by
    the lowest count found.
    """
    df = df.drop_duplicates()
    df = df.dropna()
    return reduce(df)


def reduce(df):
    """ Reduces all languages and genders by an equal amount
    """
    grouped = distribution(df)
    lowest_value = int(grouped.min().min()) 
    return df.groupby(['lang', 'gender'])[['noun', 'gender', 'lang']].sample(n=lowest_value)


def distribution(df: pd.DataFrame)-> pd.DataFrame:
    """ Groups values by gender and language, counts and gives a total for each
    """
    return df.groupby(['gender','lang']).size().unstack()


def get_x_y_from(df: pd.DataFrame)-> pd.DataFrame:
    """ Retrieves the 'noun' as X and 'gender'
    as y values from a DataFrame.
    """
    return df['noun'].tolist(), df['gender'].tolist()


def verify_args_are_valid(args: argparse, df: pd.DataFrame)-> bool:
    """ Checks whether all arguments are found within the possible
    languages and models available on this program.
    """
    all_args = [elem for arg_value in vars(args).values() for elem in arg_value]
    possibilities = possible_options(df)
    return all(lang in possibilities for lang in all_args)


def possible_options(df: pd.DataFrame)-> List[str]:
    """ Retrieves all possible options for languages
    and models that are found on this program.
    """
    languages = df['lang'].unique().tolist()
    models_ = get_classes_from_models(models)
    return languages + models_

def get_classes_from_models(custom_models):
    all_members = inspect.getmembers(custom_models)
    return [member[1].__name__ for member in all_members if inspect.isclass(member[1]) and member[1].__module__== 'models']


def get_pretrained_file(args: List, model: str)-> str:
    """ Creates a path to a potential pretrained
    file within the /saved_models directory.
    """
    langs = "-".join([lang for lang in args])
    folder_path = 'saved_models'
    file_name = f"{model}_trained_on_{langs}.pth"
    return Path(folder_path) / file_name


def build_dataloader(X, y, batch_size, max_length, clf):
    ds = models.NounDataset(X, y, clf.tokenizer, max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
    

def save_metadata(results, model_name, args):
    meta = {
        'model': model_name,
        'trained_on': args.train,
        'evaluated_on': args.evaluate,
        'accuracy': results
    }
    folder_path = 'results'
    file_name = f"{model_name}_trained_on_{args.train}_evaluated_on{args.evaluate}.json"
    full_name =  Path(folder_path) / file_name
    with open(full_name, 'w') as f:
        json.dump(meta, f)
    print(f"metadata was successfully saved as {full_name}")