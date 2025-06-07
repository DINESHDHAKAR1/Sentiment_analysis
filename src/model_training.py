import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import os

class TweetSentimentDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = self.preprocessor.text_to_indices(text)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class TweetSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, hidden = self.gru(embedded)
        hidden = self.dropout(hidden[-1])
        output = self.fc(hidden)
        return output

class ModelTrainer:
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, output_dim=3, n_layers=2, dropout=0.1, log_dir="logs"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TweetSentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.logger = logging.getLogger("ModelTrainer")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "model_training.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger.info(f"Device: {self.device}")

    def train(self, train_data, preprocessor, batch_size=32, epochs=3):
        try:
            dataset = TweetSentimentDataset(train_data["tweet"].values, train_data["sentiment"].values, preprocessor)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for i, (inputs, labels) in enumerate(data_loader, 1):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    if i % 5 == 0:
                        self.logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(data_loader)}, Loss: {loss.item():.3f}")
                avg_loss = total_loss / len(data_loader)
                self.logger.info(f"Average training loss of epoch # {epoch+1}: {avg_loss:.3f}")
            return self.model
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def save_model(self, model_path):
        try:
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise