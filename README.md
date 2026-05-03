# Car Price Prediction (Deep Learning)

**Regression | Artificial Neural Networks (ANN) | TensorFlow**

## Project Overview
This project predicts automobile prices based on technical specifications (horsepower, engine size, curb weight, etc.) using a Deep Learning approach. 

## Model Architecture
I designed a Multi-layer Perceptron (MLP) with:
- **Input Layer:** Scaled features from the Automobile dataset.
- **Hidden Layers:** 3 Dense layers (128, 64, 32 neurons) with ReLU activation.
- **Regularization:** Dropout (20%) to prevent overfitting.
- **Optimization:** Adam optimizer with adaptive learning rate.

## Tech Stack
* **Framework:** TensorFlow / Keras
* **Library:** Scikit-Learn, Pandas, Matplotlib, Seaborn.
* **Techniques:** StandardScaler, One-Hot Encoding, Early Stopping.

## Performance
The model achieves a high **R2 Score**, demonstrating its ability to capture non-linear trends in car pricing. 

## Files
- `car_price_prediction.py`: Training script.
- `car_price_model.h5`: Saved Keras model.
- `scaler_ann.pkl`: Scaler for data normalization.

## Result
<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/9147f7d9-f277-4ef9-8288-38200390c423" />

