import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import logging
import os
from .model_training import TweetSentimentDataset

class ModelEvaluator:
    def __init__(self, model, preprocessor, log_dir="logs"):
        self.model = model
        self.preprocessor = preprocessor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logging.getLogger("ModelEvaluator")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "model_evaluation.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def evaluate(self, test_data, batch_size=32):
        try:
            dataset = TweetSentimentDataset(test_data["tweet"].values, test_data["sentiment"].values, self.preprocessor)
            data_loader = DataLoader(dataset, batch_size=batch_size)
            self.model.eval()
            total_loss = 0
            predicted_labels = []
            true_labels = []

            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    _, predictions = torch.max(outputs, dim=1)
                    predicted_labels.extend(predictions.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

            avg_loss = total_loss / len(data_loader)
            accuracy = (torch.tensor(predicted_labels) == torch.tensor(true_labels)).sum().item() / len(true_labels)
            report = classification_report(true_labels, predicted_labels, target_names=["positive", "neutral", "negative"])
            cm = confusion_matrix(true_labels, predicted_labels)
            self.logger.info(f"Evaluation results: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}\nClassification Report:\n{report}")
            return {
                "loss": avg_loss,
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": cm
            }
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise