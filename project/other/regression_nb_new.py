import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error, mean_squared_log_error
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv('linearFirstOrder100.csv')  # Update with your file path

# Separate features (inputs) and target (outputs)
X = df[["SSE%", "SSE", "rise_time"]]
y = df[["K", "lam1", "lam2", "w1", "w2"]]

# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Define models for comparison
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': GridSearchCV(Ridge(), {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}, cv=5, scoring='neg_mean_squared_error'),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Polynomial Regression (Degree 2)': Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
    'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42),
    'XGBoost Regressor': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

# Add a custom deep learning model using Keras
def build_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

# Add deep learning model to models dictionary
models['Neural Network (Keras)'] = build_nn_model(X_train.shape[1])

# Initialize results dictionary to store metrics for each model
results = {}

# Evaluate each model
for model_name, model in models.items():
    if model_name == 'Neural Network (Keras)':
        # Fit the Keras model
        history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0, validation_split=0.2)
        y_pred = model.predict(X_test)
        training_loss = history.history['loss'][-1]
        validation_loss = history.history['val_loss'][-1]
    elif model_name == 'Ridge Regression':
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
        y_pred = best_model.predict(X_test)
        training_loss = -model.best_score_
        validation_loss = None  # GridSearchCV does not provide direct validation loss
    else:
        # Fit other models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        training_loss = mean_squared_error(y_train, model.predict(X_train))
        validation_loss = mean_squared_error(y_test, y_pred)
    
    # Inverse transform the scaled outputs for interpretation
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    median_ae = median_absolute_error(y_test_original, y_pred_original)
    ev_score = explained_variance_score(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    try:
        msle = mean_squared_log_error(y_test_original, y_pred_original)
    except ValueError:
        msle = 'N/A (Negative values in data)'
    
    # Store results
    results[model_name] = {
        'Training Loss': training_loss,
        'Validation Loss': validation_loss,
        'Mean Squared Error (MSE)': mse,
        'Mean Absolute Error (MAE)': mae,
        'Median Absolute Error': median_ae,
        'Mean Squared Log Error (MSLE)': msle,
        'Explained Variance Score': ev_score,
        'R^2 Score': r2
    }

# Print out results for comparison
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")

# Feature importance for Random Forest
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    rf_model.fit(X_train, y_train)
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\nFeature Importance (Random Forest):\n", importance_df)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance from Random Forest Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Save the output to a CSV file
output_df = pd.DataFrame({
    'Predicted_K': y_pred_original[:, 0],
    'Actual_K': y_test_original[:, 0],
    'Predicted_lam1': y_pred_original[:, 1],
    'Actual_lam1': y_test_original[:, 1],
    'Predicted_lam2': y_pred_original[:, 2],
    'Actual_lam2': y_test_original[:, 2],
    'Predicted_w1': y_pred_original[:, 3],
    'Actual_w1': y_test_original[:, 3],
    'Predicted_w2': y_pred_original[:, 4],
    'Actual_w2': y_test_original[:, 4]
})
output_df.to_csv('model_predictions_output.csv', index=False)

# Visualize model results - Plotting predicted vs actual values for each output variable (using the last evaluated model)
plt.figure(figsize=(18, 10))

# Plot for K
plt.subplot(2, 3, 1)
plt.scatter(y_test_original[:, 0], y_pred_original[:, 0], alpha=0.6)
plt.xlabel('Actual K')
plt.ylabel('Predicted K')
plt.title('Actual vs Predicted K')
plt.plot([min(y_test_original[:, 0]), max(y_test_original[:, 0])], [min(y_test_original[:, 0]), max(y_test_original[:, 0])], color='red', linestyle='--')

# Plot for lam1
plt.subplot(2, 3, 2)
plt.scatter(y_test_original[:, 1], y_pred_original[:, 1], alpha=0.6)
plt.xlabel('Actual lam1')
plt.ylabel('Predicted lam1')
plt.title('Actual vs Predicted lam1')
plt.plot([min(y_test_original[:, 1]), max(y_test_original[:, 1])], [min(y_test_original[:, 1]), max(y_test_original[:, 1])], color='red', linestyle='--')

# Plot for lam2
plt.subplot(2, 3, 3)
plt.scatter(y_test_original[:, 2], y_pred_original[:, 2], alpha=0.6)
plt.xlabel('Actual lam2')
plt.ylabel('Predicted lam2')
plt.title('Actual vs Predicted lam2')
plt.plot([min(y_test_original[:, 2]), max(y_test_original[:, 2])], [min(y_test_original[:, 2]), max(y_test_original[:, 2])], color='red', linestyle='--')

# Plot for w1
plt.subplot(2, 3, 4)
plt.scatter(y_test_original[:, 3], y_pred_original[:, 3], alpha=0.6)
plt.xlabel('Actual w1')
plt.ylabel('Predicted w1')
plt.title('Actual vs Predicted w1')
plt.plot([min(y_test_original[:, 3]), max(y_test_original[:, 3])], [min(y_test_original[:, 3]), max(y_test_original[:, 3])], color='red', linestyle='--')

# Plot for w2
plt.subplot(2, 3, 5)
plt.scatter(y_test_original[:, 4], y_pred_original[:, 4], alpha=0.6)
plt.xlabel('Actual w2')
plt.ylabel('Predicted w2')
plt.title('Actual vs Predicted w2')
plt.plot([min(y_test_original[:, 4]), max(y_test_original[:, 4])], [min(y_test_original[:, 4]), max(y_test_original[:, 4])], color='red', linestyle='--')

plt.tight_layout()
plt.show()