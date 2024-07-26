# Forecast Model API

## Introduction
This API serves a trained model to predict sales based on date, store, and item.

## How to Build the Docker image
   ```sh
   cd app
   docker build -t forecast_api .
   ```
## How to Deploy

1. Clone the repository.
2. Download the datasets from Kaggle and place them in the `data` directory.
3. Build and train the model:

   ```sh
   python model/train_model.py
   ```
4. Train the Model with hyperparameter tuning:

   ```sh
   python model/train_model.py --tune
   ```

## How to Run the Docker container
   ```sh
   docker run -p 5000:5000 forecast_api
   ```

## Endpoints

- `/predict`: Given JSON input, returns the predicted sales.
- `/status`: Returns the API status.

## Predict Sales
   ```sh
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"date": "2013-01-01", "store": 1, "item": 1}'
   ```

## Check Status
   ```sh
   curl http://localhost:5000/status
   ```