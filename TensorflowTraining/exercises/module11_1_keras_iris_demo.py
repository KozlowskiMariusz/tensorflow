# Module 8 Keras
# Step 1: Get Data

import numpy as np
from sklearn import datasets
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from keras.models import Sequential

iris = datasets.load_iris()
data = iris["data"]
target = iris["target"]

# Step 2: Preprocess Data - One Hot Encoding
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!

# Step 3: Shuffle/Spliut Data

train_X, test_X, train_y, test_y = train_test_split(data, all_Y, test_size=0.33, random_state=42)

# Step 4: Model


model = Sequential()


model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# Step 5: Loss and Optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 6 Training
model.fit(train_X, train_y, epochs=150, batch_size=10)

# Step 7: Metrics
loss_and_metrics = model.evaluate(test_X, test_y, batch_size=128)