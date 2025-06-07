import pandas as pd
import logging
import os

class DataIngestion:
    def __init__(self, data_path, log_dir="logs"):
        self.data_path = data_path
        self.logger = logging.getLogger("DataIngestion")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "data_ingestion.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def load_data(self):
        try:
            data = pd.read_csv(self.data_path)
            if 'tweet' not in data.columns or 'sentiment' not in data.columns:
                raise ValueError("CSV must contain 'tweet' and 'sentiment' columns")
            self.logger.info(f"Successfully loaded data from {self.data_path}")
            freq=data['sentiment'].value_counts()
            self.logger.info(f"frequency of classes:{freq}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise