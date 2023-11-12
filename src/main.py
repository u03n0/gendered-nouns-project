import torch
import numpy as np
from torch.utils.data import  DataLoader, SubsetRandomSampler
from models import LanguageDataset, GenderedLSTM, GenderedCNN
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Check if you have a GPU: 
print(torch.cuda.is_available())

if __name__ == "__main__":
    with open("french_clean.txt") as f:
        text = f.readlines()

    # Hyperparameters for LSTM model
    lstm_vocab_size = 70
    lstm_window_size = 10
    lstm_embedding_dim = 2
    lstm_hidden_dim = 16
    lstm_dense_dim = 32
    lstm_n_layers = 1
    lstm_max_norm = 2

    # Hyperparameters for CNN model
    cnn_vocab_size = 70
    cnn_embedding_dim = 2
    cnn_num_filters = 100
    cnn_filter_sizes = [3, 4, 5] 
    cnn_output_dim = 1  # Binary classification
    cnn_dropout = 0.5

    # Training config
    train_val_split = 0.8
    batch_size = 128
    random_state = 13

    torch.manual_seed(random_state)

    # Creating datasets
    dataset = LanguageDataset(text)
    n_samples = len(dataset)
    split_idx = int(n_samples * train_val_split)
    
    n_epochs = dataset.longest_noun    

    train_indices, val_indices = np.arange(split_idx), np.arange(split_idx, n_samples)

    train_dataloader = DataLoader(
                dataset, sampler=SubsetRandomSampler(train_indices), batch_size=batch_size
        )
    val_dataloader = DataLoader(
                dataset, sampler=SubsetRandomSampler(val_indices), batch_size=batch_size
    )

    # Creating and training LSTM model
    # lstm_model = GenderedLSTM(train_dataloader, lstm_embedding_dim, lstm_hidden_dim, lstm_vocab_size, 2)
    # lstm_model.train(train_dataloader, val_dataloader, 20, batch_size, device='cuda')

    # Creating and training CNN model
    cnn_model = GenderedCNN(cnn_vocab_size, cnn_embedding_dim, cnn_num_filters, cnn_filter_sizes, cnn_output_dim, cnn_dropout)
    cnn_model.train_cnn(train_dataloader, val_dataloader, 20, batch_size, device='cpu', binary=True) # Two genders for French
   
    # TDODO: Never used a gradCAM before. Should be figured out:
    # Apply Grad-CAM to a sample text
    sample_text = "your_sample_text_here"  # Replace with your actual sample text
    sample_input = dataset.__getitem__(0)  # Assuming you want to visualize the first sample in the dataset
    sample_input_text = sample_input[0]  # Assuming the input is a text (modify if necessary)

    # Convert the sample text to a tensor
    transform = transforms.Compose([transforms.ToTensor()])
    sample_input_tensor = transform(sample_input_text)

    # Obtain Grad-CAM
    gradcam = cnn_model.get_gradcam(sample_input_tensor.unsqueeze(0), target_class=0)  # Assuming binary classification

    # Display the original input and Grad-CAM side by side
    plt.figure(figsize=(10, 5))

    # Original input
    plt.subplot(1, 2, 1)
    plt.title('Original Input')
    plt.imshow(transforms.ToPILImage()(sample_input_tensor.squeeze()))

    # Grad-CAM
    plt.subplot(1, 2, 2)
    plt.title('Grad-CAM')
    plt.imshow(gradcam.squeeze(), cmap='jet', alpha=0.5)
    plt.imshow(transforms.ToPILImage()(sample_input_tensor.squeeze()), alpha=0.5)

    plt.show()