import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt

efa_df = pd.read_csv("2014_2025_EFA_Monthly.csv", parse_dates=['Date'], index_col='Date')
fund_df = pd.read_csv("2014_2025_dataset_Monthly.csv", parse_dates=['Date'], index_col='Date')

efa_df['Performance'] = efa_df['Performance'].astype(str).str.replace('%', '').astype(float) / 100
fund_df['Performance'] = fund_df['Performance'].astype(str).str.replace('%', '').astype(float) / 100

df = pd.merge(efa_df, fund_df, left_index=True, right_index=True, suffixes=('_EFA', '_Fund'))

window_size = 6
X, y = [], []

for i in range(len(df) - window_size):
    X.append(df['Performance_EFA'].iloc[i : i + window_size].values)
    y.append(df['Performance_Fund'].iloc[i + window_size])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(X, y, epochs=50, verbose=0)

predicted_systematic = model.predict(X).flatten()
actual_returns = y

cnn_alpha = actual_returns - predicted_systematic

plt.figure(figsize=(12, 6))
plt.plot(df.index[window_size:], actual_returns, label='Actual Fund Return', color='blue', alpha=0.5)
plt.plot(df.index[window_size:], predicted_systematic, label='CNN Predicted (EFA-Driven / Beta)', color='orange', linewidth=2)
plt.bar(df.index[window_size:], cnn_alpha, label='Idiosyncratic Return (Alpha)', color='green', alpha=0.4)
plt.title('CNN Return Decomposition: Actual vs. EFA-Driven vs. Alpha')
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.show()