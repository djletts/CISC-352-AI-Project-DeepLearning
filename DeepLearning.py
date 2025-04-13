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

trials = 5000
n=5 # Number of rows and columns in the grid

#hyper pearameters variables
layer1Size=512
layer2Size=256  
layer3Size=128

# Generate 5000 random nonograms
for i in range(trials):
    # Generate a random string of 25 1s and 0s
    string = ''
    for j in range(n**2):
        string += str(random.randint(0, 1)) 
    grid = [[] for i in range(n)] # Create a 5x5 grid

    # Split the string into a 5x5 grid
    for i in range(len(string)):
        grid[i%n].append(int(string[i]))
    list.append(grid)


finalList = []

# Create the keys for the rows and columns
for grid in list:
    rowList = []
    colList = []

    for i in range(n):
        # Create the keys for the rows
        keyR = []
        countR = 0
        for j in range(n):
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
        for j in range(n):
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
    xSum=[]
    ySum=[]
    for i in range(n):
        ySum += item[0][i]
        xSum += item[1][i] + item[2][i]

    y.append(ySum)
    X.append(xSum)
X = np.array(X)
y = np.array(y)

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(layer1Size, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(layer2Size, activation='relu'),
    tf.keras.layers.Dense(layer3Size, activation='relu'),
    tf.keras.layers.Dense(n**2, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=250, batch_size=2000, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')
print(f'Test loss: {loss}')

# Predict the results for testing
predictions = model.predict(X_test)
correct_count = 0
for i in range(len(predictions)):

    if all(predictions[i].round(0, None) == y_test[i]):
        correct_count += 1
print(f'Correct predictions: {correct_count} out of {len(predictions)}')
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

