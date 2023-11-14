import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, BertForSequenceClassification


class LanguageDataset(Dataset):
    """ Custom Language dataset 
    
    """

    def __init__(self, data):
        self.data = data
        self.char2idx = {}
        self.labels = []
        self.longest_noun = 0

        idx = 0
        for tup in data:
            noun, label = tup.strip().split(",")
            if len(noun) > self.longest_noun:
                self.longest_noun = len(noun)
            for char in noun:
                if char not in self.char2idx:
                    self.char2idx[char] = idx
                    idx +=1
            if label not in self.labels:
                self.labels.append(label)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pass

class NounDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        self.X = X
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(y)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = self.X[idx]
        label = torch.tensor(self.y[idx])

        encoded_text = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return  {
        'input_ids': encoded_text['input_ids'].squeeze(),
        'attention_mask': encoded_text['attention_mask'].squeeze(),
        'label': label
    }

class GenderedLSTM(nn.Module):

    def __init__(self, traingenerator, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(GenderedLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.allocate_params(traingenerator)


    def allocate_params(self, datagenerator):

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)


    def foward(self, sequence):
        embeds = self.word_embeddings(sequence)
        lstm_out, _ = self.lstm(embeds.view(len(sequence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sequence), -1))
        tag_score = F.log_softmax(tag_space, dim=1)
        return tag_score
    
    def train_lstm(self, traingenerator, validgenerator, epochs, batch_size, device='cuda', learning_rate=0.1):
        pass


class GenderedCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super(GenderedCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        # Fully connected layer for final classification
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def foward(self, text):
        # Embedding layer
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1) # Adds a channel/ feature map dimension for the conv layers

        # Convolutional and pooling layers
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)
    
    def train_cnn(self, train_dataloader, val_dataloader, epochs, batch_size, device='cuda', learning_rate=0.001, binary=True):
        self.to(device)
        
        # Choose the appropriate loss function based on binary or multiclass classification
        if binary: criterion = nn.BCELoss() 
        else: criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            # Training
            self.train()
            total_loss = 0.0
            correct = 0
            total_samples =0

            for data in train_dataloader:
                inputs, labels = datainputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().items()
                total_samples += labels.size(0)

            train_accuracy = correct / total_samples
            avg_loss = total_loss / len(train_dataloader)

            # Validation
            self.eval()
            val_loss = 0.0
            correct = 0
            total_samples = 0

            with torch.no_grad():
                for data in val_dataloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.mzx(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

            val_accuracy = correct / total_samples
            avg_val_loss = val_loss / len(val_dataloader)

            print(f'Epoch {epoch + 1}/{epochs}, '
                  f'Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, '
                  f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


    def get_gradcam(self, input_tensor, target_class):
        # Set up hooks to capture intermediate activations and gradients
        activations = []
        gradients = []

        def foward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        # Register hooks on the last convolutional layer
        hook_handles = []
        target_layer = None

        for layer in reversed(self.features.childent()):
            if isinstance(layer, nn.Conv2d):
                target_layer = layer
                break
        
        hook_handles.append(target_layer.register_forward_hook(foward_hook))
        hook_handles.append(target_layer.register_backward_hook(backward_hook))

        # Foward pass
        logits = self(input_tensor)

        # Backward pass to compute gradients
        logits.backward()

        # Remove hooks
        for handle in hook_handles:
            handle.remove()

        # Compute Grad-CAM
        gradients = gradients[0] # Take the first element of the list (the only one)
        activations = activations[0] # Take the first element of the list

        weights = torch.mean(gradients, dim=(2,3), keepdim=True)
        gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
        gradcam = F.relu(gradcam)

        # Resize Grad-CAM to match the input size
        gradcam = F.interpolate(gradcam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        # Normalize to [0, 1]
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())

        return gradcam.squeeze()


# BERT 
class GenderBert(nn.Module):
    def __init__(self, **kwargs):
        super(GenderBert, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=kwargs.get("epochs"))

    def train_model(self, train_loader, device, num_epochs=3):
        self.model.to(device) 
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].squeeze().long().to(device)
                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    def evaluate(self, data_loader, device, mode='test'):
        self.model.to(device)
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'{mode.capitalize()} Accuracy: {accuracy * 100:.2f}%')
        return accuracy

