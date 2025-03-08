import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv()

# Split the data into initial and goal
X = data[['col0', 'col1', 'col2', 'col3', 'col4', 'row0', 'row1', 'row2', 'row3', 'row4']]
y = data[['0, 0', '0, 1', '0, 2', '0, 3', '0, 4', '1, 0', '1, 1', '1, 2', '1, 3', '1, 4', '2, 0', '2, 1', '2, 2', '2, 3', '2, 4', '3, 0', '3, 1', '3, 2', '3, 3', '3, 4', '4, 0', '4, 1', '4, 2', '4, 3', '4, 4']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(25)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)
print(f"Test Mean Square Error: {test_loss}")
print(f"Test Mean Absolute Error: {test_mae}")

    
