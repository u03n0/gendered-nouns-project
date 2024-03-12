import torch

import pandas as pd

from sklearn.model_selection import train_test_split



if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
