# Order Classification with LLaMA And HoeffdingTreeClassifier Model Combinedly

A hybrid machine learning project designed to detect fake orders by combining the power of natural language processing (NLP) and incremental learning. This project uses Hugging Face's LLaMA model for text feature extraction and HoeffdingTreeClassifier for real-time classification of orders.

## Features

- **Text Analysis**: Leverages LLaMA to extract meaningful features from order-related text, such as customer names, event descriptions, and emails.
- **Fraud Detection**: Uses HoeffdingTreeClassifier to classify orders as fake or genuine in real-time.
- **Scalable**: Suitable for high-throughput applications with incremental learning capabilities.
- **Customizable**: Flexible architecture to adapt to new features or datasets.

## Use Case

- Detecting fake bookings in event management systems or eCommerce platforms.

## Tech Stack

- **Hugging Face Transformers (LLaMA)**: NLP feature extraction.
- **Scikit-multiflow**: Implementation of HoeffdingTreeClassifier for real-time classification.
- **Python**: Primary language for development and integration.

## Get Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/TanvirIslam-BD/fake-order-detector.git
   
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   - Add your dataset to the `data/` folder in CSV format. Ensure it contains necessary features such as event descriptions, emails, etc.

4. **Train the Model**:
   ```bash
   python train_model.py
   ```

5. **Run Predictions**:
   ```bash
   python predict.py --input sample_order.csv
   ```


## To Startup cron
```
 celery -A django_ml_app beat --loglevel=debug

```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the [MIT License](LICENSE).


