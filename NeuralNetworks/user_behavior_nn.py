import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import keras
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv('user_behavior_dataset.csv')

# One-hot encoding for categorical variables
columns_to_encode = [col for col in ["Gender", "Device Model", "Operating System"] if col in df.columns]
df = pd.get_dummies(df, columns=columns_to_encode, dtype="int64", drop_first=True)

# Display dataset structure
print(df.columns)
print(df.head())

# Correlation matrix
corr = df.corr(numeric_only=True)

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Select features and target
x = df[['App Usage Time (min/day)', 'Battery Drain (mAh/day)',
        'Number of Apps Installed', 'Data Usage (MB/day)', 'User Behavior Class']]
y = df['Screen On Time (hours/day)']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25, shuffle=True)

# Define neural network model
def build_model(my_learning_rate, num_features):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=1, input_shape=(num_features,)))
    model.compile(optimizer=Adam(my_learning_rate), loss='mean_squared_error', metrics=['mse', 'mae'])
    return model

# Train model
def train_model(model, features, label, epochs, batch_size):
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs)
    return history

# Plot training loss curve
def plot_the_loss(hist, title):
    plt.figure(figsize=(8, 6))
    plt.plot(hist.history['loss'], label='Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Experiment 1: Single Feature (App Usage Time)
model = build_model(0.001, 1)
history = train_model(model, x_train[['App Usage Time (min/day)']], y_train, epochs=100, batch_size=10)
plot_the_loss(history, "Training Curve: Single Feature (App Usage Time)")

# Prediction and evaluation for Experiment 1
def pred(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print('R² score:', r2_score(y_test, y_pred))
    print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
    print('MAE:', mean_absolute_error(y_test, y_pred))
    return y_pred

y_pred = pred(model, x_test[['App Usage Time (min/day)']], y_test)

# Experiment 2: All Features
model = build_model(0.001, x_train.shape[1])
history = train_model(model, x_train, y_train, epochs=400, batch_size=10)
plot_the_loss(history, "Training Curve: All Features")

# Prediction and evaluation for Experiment 2
y_pred = pred(model, x_test, y_test)

# Display some predictions and actual values
print("First 5 Predictions:", y_pred[:5].flatten())
print("First 5 Actual Values:", y_test.head().values)

# Test prediction for a new user
new_user = pd.DataFrame([[200, 1500, 350, 300, 100]], 
                        columns=['App Usage Time (min/day)', 'Battery Drain (mAh/day)', 
                                 'Number of Apps Installed', 'Data Usage (MB/day)', 'User Behavior Class'])
new_user_scaled = StandardScaler().fit_transform(new_user)
new_prediction = model.predict(new_user_scaled)
print(f'Predicted Screen On Time for new user: {np.round(new_prediction[0], 2)} hours/day')


