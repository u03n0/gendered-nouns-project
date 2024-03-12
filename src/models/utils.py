import torch
import csv 


def reverse_sequence(noun):
    return noun[::-1]


def get_data_from_df(df):
    nouns = df.iloc[:,0].tolist()
    gender = df.iloc[:,1].tolist()   
    noun_chars = [[char for char in noun] for noun in nouns]
    return noun_chars, gender


def save_probabilities(model_checkpoint, filename):
    checkpoint = torch.load(model_checkpoint)
    train = checkpoint['train_char_prediction_probs']
    valid = checkpoint['valid_char_prediction_probs']

    # Sorting the words in alphabetical order
    sorted_train = dict(sorted(train.items()))
    sorted_valid = dict(sorted(valid.items()))

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for train_word, train_pred_probs in sorted_train.items():
            writer.writerow([train_word, train_pred_probs])
        for valid_word, valid_pred_probs in sorted_valid.items():
            writer.writerow([valid_word, valid_pred_probs])
        print(f'File successfully written to {filename}.')
    return

def save_padded_words(filename, batch_of_words):
    lines = ['\t'.join(word) + '\n' for word in batch_of_words]
    with open(filename, 'a', encoding='utf-8') as f:
        f.writelines(lines)
    return 