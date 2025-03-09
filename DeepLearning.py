import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# For the dataset, we will generate 5000 random nonograms
# Each nonogram will have a 5x5 grid
list = []
for i in range(5000):
    # Generate a random string of 25 1s and 0s
    string = ''
    for j in range(25):
        string += str(random.randint(0, 1)) 
    grid = [[], [], [], [], []]

    # Split the string into a 5x5 grid
    for i in range(len(string)):
        grid[i%5].append(int(string[i]))
    list.append(grid)


finalList = []

# Create the keys for the rows and columns
for grid in list:
    rowList = []
    colList = []

    for i in range(5):
        # Create the keys for the rows
        keyR = []
        countR = 0
        for j in range(5):
            if grid[i][j] == 1:
                countR += 1
            else:
                if countR != 0:
                    keyR.append(countR)
                countR = 0
        if countR != 0:
            keyR.append(countR)
        while len(keyR) < 3:
            keyR.append(0)
        rowList.append(keyR)     

        # Create the keys for the columns
        keyC = []
        countC = 0
        for j in range(5):
            if grid[j][i] == 1:
                countC += 1
            else:
                if countC != 0:
                    keyC.append(countC)
                countC = 0
        if countC != 0:
            keyC.append(countC)
        while len(keyC) < 3:
            keyC.append(0)
        colList.append(keyC)

        finalList.append([grid, colList, rowList])
# Split the data into initial and goal
X = []
y = []

# Create the input and output data
for item in finalList:
    y.append(item[0][0] + item[0][1] + item[0][2] + item[0][3] + item[0][4])
    X.append(item[1][0] + item[1][1] + item[1][2] + item[1][3] + item[1][4] + item[2][0] + item[2][1] + item[2][2] + item[2][3] + item[2][4])
X = np.array(X)
y = np.array(y)

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(25, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=2000, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')
print(f'Test loss: {loss}')

# Predict the results for testing
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

