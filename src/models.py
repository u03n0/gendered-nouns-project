import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer, BertForSequenceClassification


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
        attention_mask = encoded_text['attention_mask'].to(torch.int)
        return  {
        'input_ids': encoded_text['input_ids'].squeeze(),
        'attention_mask': attention_mask.squeeze(),
        'label': label
    }


class Bert(nn.Module):
    def __init__(self, num_labels=2):
        super(Bert, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    def train_model(self, train_loader, device, num_epochs=3):
        self.model.to(device) 
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

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
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
        }, path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        print(f'Model loaded from {path}')
