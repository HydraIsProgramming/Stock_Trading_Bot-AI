## Stock Trading Bot with Deep Learning
Introduction
This project explores the application of deep learning techniques to predict stock prices. We developed a custom Convolutional Neural Network (CNN) and experimented with pre-trained models like ResNet50, InceptionV3, and DenseNet121. The goal is to predict stock prices based on historical market data and make informed trading decisions. This repository includes the implementation of the models, data preprocessing, evaluation, and deployment using a user-friendly interface.

## Database
1) Install Kaggle Dataset from https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
2) pip install -r requirements.txt to install all dependencies


## Project Phases
Custom CNN Model: Developed a CNN model to predict stock prices from historical market data.
Pre-Trained Models: Fine-tuned pre-trained models (ResNet50, InceptionV3, DenseNet121) for stock price prediction.
Ensemble Model: Combined predictions from multiple models using an ensemble approach.
Deployment: Created a Gradio interface for user interaction and prediction visualization.


## Pre-Trained Models
ResNet50: (https://keras.io/api/applications/resnet/#resnet50-function)

InceptionV3: https://keras.io/api/applications/inceptionv3/

DenseNet121: https://keras.io/api/applications/densenet/#densenet121-function

## Features
Stock Price Prediction: Predict future stock prices based on historical data.

Custom and Pre-Trained Models: Utilize both custom CNN and pre-trained models.

Ensemble Learning: Combine predictions from multiple models for improved accuracy.

User Interface: Interactive Gradio interface for easy user interaction.

## Installation
To get started, clone the repository and install the required packages.

Copy code

1) git clone https://github.com/your-username/stock-trading-bot.git

2) cd stock-trading-bot

3) pip install -r requirements.txt


## Data Preparation: 
Ensure that the stock data is available in the archive/stocks directory. Each stock's data should be in a separate CSV file with the stock symbol as the filename.

## Running the Models: 
You can run the setup.ipynb Jupyter notebook to train the models and make predictions. The notebook includes detailed steps for data preprocessing, model training, and evaluation.

## Gradio Interface: 
Launch the Gradio interface for an interactive experience.

Gradio Interface

The Gradio interface allows users to:

Select a stock symbol.

Set initial capital, start, and end dates.

View predicted prices, plots, and trading decisions.

The interface also includes a chatbot for quick stock-related queries.



## Evaluation Metrics

The models were evaluated using the following metrics:


Mean Squared Error (MSE)

Mean Absolute Error (MAE)

R-squared (R²)

Confusion Matrix

ROC Curve

Precision-Recall Curve

