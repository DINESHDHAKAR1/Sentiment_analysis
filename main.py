from src import SentimentAnalysisPipeline

if __name__ == "__main__":
    pipeline = SentimentAnalysisPipeline(
        data_path="tweets.csv",
        model_path="sentiment_model.pth"
    )
    evaluation_report = pipeline.run()
    print("Evaluation Report:")
    print(f"Loss: {evaluation_report['loss']:.4f}")
    print(f"Accuracy: {evaluation_report['accuracy']:.4f}")
    print("Classification Report:\n", evaluation_report['classification_report'])
    
    sample_tweets = [
        "I love this product! It's amazing!",
        "This is the worst experience ever",
        "I like this thing",
        "never ever going to visit this place ",
        "The event starts at 5 PM."
    ]
    predictions = pipeline.predict(sample_tweets)
    print("Sample Predictions:", predictions)