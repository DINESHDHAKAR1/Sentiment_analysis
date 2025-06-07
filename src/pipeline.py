import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
from .data_ingestion import DataIngestion
from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator
import torch

class SentimentAnalysisPipeline:
    def __init__(self, data_path, model_path, log_dir="logs"):
        self.data_path = data_path
        self.model_path = model_path
        self.logger = logging.getLogger("SentimentAnalysisPipeline")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "sentiment_pipeline.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.data_ingestion = DataIngestion(data_path, log_dir)
        self.preprocessor = DataPreprocessor(log_dir)
        self.model_trainer = None
        self.model_evaluator = None

    def run(self):
        try:
            # Load and preprocess data
            data = self.data_ingestion.load_data()
            data = self.preprocessor.preprocess(data)
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            self.logger.info("Data split into training and testing sets")

            # Initialize and train model
            self.model_trainer = ModelTrainer(
                vocab_size=len(self.preprocessor.word_to_index),
                embedding_dim=100,
                hidden_dim=256,
                output_dim=3,
                n_layers=2,
                dropout=0.1,
                log_dir="logs"
            )
            self.model_trainer.train(train_data, self.preprocessor)
            self.model_trainer.save_model(self.model_path)
            self.logger.info(f"Model training completed and saved to:{self.model_path}")

            # Evaluate model
            self.model_evaluator = ModelEvaluator(self.model_trainer.model, self.preprocessor)
            evaluation_report = self.model_evaluator.evaluate(test_data)
            self.logger.info("Pipeline execution completed")
            return evaluation_report
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def predict(self, texts, model_path="/home/dinesh/Documents/GitHub/Sentiment_analysis/sentiment_model.pth"):
        try:
            if isinstance(texts, str):
                texts = [texts]

            processed_texts = [self.preprocessor.clean_text(text) for text in texts]
            indices = [self.preprocessor.text_to_indices(text) for text in processed_texts]
            inputs = torch.tensor(indices, dtype=torch.long).to(self.model_trainer.device)

            
            self.model_trainer.model.load_state_dict(torch.load(model_path, map_location=self.model_trainer.device))
            self.model_trainer.model.eval()

            with torch.no_grad():
                outputs = self.model_trainer.model(inputs)
                _, predictions = torch.max(outputs, dim=1)

            
            label_map = {0: "positive", 1: "neutral", 2: "negative"}
            predictions = [label_map[pred.item()] for pred in predictions]

            self.logger.info(f"Predictions made for {len(texts)} texts")
            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
