import matplotlib.pyplot as plt
import numpy as np
from math import *
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# For the dataset, we will generate 5000 random nonograms
# Each nonogram will have a 5x5 grid
list = []

trials = 5000
n = int(input("Enter the number of columns/rows you want(columns=rows since grid is a square): "))
epoch = int(input("Enter the number of epochs you want: "))
 # Number of rows and columns in the grid

#hyper pearameters variables
layer1Size=512
layer2Size=256  

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
        while len(keyR) < ceil(n/2):
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
        while len(keyC) < ceil(n/2):
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

def accuracy(y_true, y_pred):
    y_true = tf.keras.backend.cast(y_true, tf.keras.backend.floatx())
    y_pred = tf.keras.backend.cast(y_pred, tf.keras.backend.floatx())
    return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.round(y_true), tf.keras.backend.round(y_pred)))


# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(layer1Size, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(layer2Size, activation='relu'),
    tf.keras.layers.Dense(n**2, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[accuracy])

# Train the model
history = model.fit(X_train, y_train, epochs=epoch, batch_size=2000, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')
print(f'Test loss: {loss}')

# Predict the results for testing
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
plt.figure(0)
plt.plot(history.history['val_accuracy'])
plt.ylabel('validation accuracy')
plt.xlabel('epoch')
plt.figure(1)
plt.plot(history.history['val_loss'])
plt.ylabel('validation loss')
plt.xlabel('epoch')
plt.show()