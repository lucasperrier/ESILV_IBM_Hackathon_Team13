# ESILV IBM Hackathon - Team 13

Fault detection and classification system using machine learning for time-series sensor data.

## Demo Video

The video is accessible there :

https://youtu.be/ofiWijedRT4

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data
python main_prepare_data.py

# Train models
python main_train_classical.py

# Run web app
streamlit run app.py
```

## Download Data

- Download the following data on this webpage : https://www.sciencedirect.com/science/article/pii/S235234092500321X
- Put them in the following folder "data/raw"
- Then run main_prepare_data.py"

## Features

- **Fault Detection**: Binary classification (fault vs. no fault)
- **Fault Type Classification**: Crack, edge cut, surface cut
- **Severity Prediction**: Fault severity estimation
- **Multi-Task Learning**: CNN-based deep learning models
- **GPU Acceleration**: XGBoost and KNN with CUDA support

## Models

- **Classical ML**: XGBoost with time/frequency domain features
- **Deep Learning**: Multi-task CNN (1D convolutions)
- **KNN**: GPU-accelerated k-nearest neighbors

## Project Structure

- `data/`: Data loading and preprocessing
- `features/`: Time and frequency domain feature extraction
- `models/`: Model implementations
- `training/`: Training and evaluation utilities
- `app.py`: Streamlit web interface

## Team 13

Akram Hanouz, Branly Auriol Nkenmegny Djime, Jules Jouvin, Lucas Perrier, Marwan Bennis, Zakaria Jalal
