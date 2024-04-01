# %%
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# %%
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
df = pd.read_parquet('run1.parquet',dtype_backend = 'pyarrow')[['X_jets','y']]

# %%
# Set device to cuda
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %%
df = df.iloc[:500]

# %%
# Convert to numpy arrays
X = np.stack(df['X_jets'].values)
y = df['y'].values


# %%
print(X.shape, y.shape)

# %%
df = None

# %%
# Separate into training, validation, and test data

testLen = 0.2*len(X)
X_train = X[:int(len(X)-testLen)]
y_train = y[:int(len(X)-testLen)]

X_test = X[int(len(X)-testLen):]
y_test = y[int(len(X)-testLen):]

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# %%


# %%
# Separate into training and validation data
valLen = 0.2*len(X_train)
print(valLen)

X_valid = X_train[:int(valLen)]
y_valid = y_train[:int(valLen)]

print(X_valid.shape, y_valid.shape)

X_train = X_train[int(valLen):]
y_train = y_train[int(valLen):]

print(X_train.shape, y_train.shape)

# %%
print(X_train.shape, X_valid.shape, X_test.shape)
print(y_train.shape, y_valid.shape, y_test.shape)

# %%
# Take transpose of X data
X_train = np.transpose(X_train, (0,2,3,1))
X_test = np.transpose(X_test, (0,2,3,1))
X_valid = np.transpose(X_valid, (0,2,3,1))

print(X_train.shape, X_valid.shape, X_test.shape)

# %%
# Normalizing the data
mean = np.mean(X_train, axis=(0,1,2,3))
std = np.std(X_train, axis=(0,1,2,3))

# %%
X_train = (X_train - mean) / (std + 1e-7)
X_valid = (X_valid - mean) / (std + 1e-7)
#X_test = (X_test - mean) / (std + 1e-7)

# %%
# Import necessary libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import regularizers, optimizers


# %%
# Build Model Structure (Modified Version Of Alexnet)
base_hidden_units = 125
weight_decay = 1e-4
model = Sequential()

# %%


# %%
# from skimage.transform import resize

# # Resize all images to 32,32 in X_train
# X_train = np.array([resize(image, (128,128)) for image in X_train])
# X_valid = np.array([resize(image, (128,128)) for image in X_valid])
# X_test = np.array([resize(image, (128,128)) for image in X_test])

print(X_train.shape, X_valid.shape, X_test.shape)

# %%
print(y_train.shape, y_valid.shape, y_test.shape)

# %%
from keras.utils import to_categorical
y_train = to_categorical(y_train, 2)
y_valid = to_categorical(y_valid, 2)
y_test = to_categorical(y_test, 2)

# %%


# %%
# Convolutional Layer 1
model.add(Conv2D(base_hidden_units, kernel_size=(3,3),padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Convolutional Layer 2
model.add(Conv2D(base_hidden_units,kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Pool + Dropout
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Convolutional Layer 3
model.add(Conv2D(base_hidden_units*2, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#Convolutional Layer 4
model.add(Conv2D(base_hidden_units*2, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Pool + Dropout
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

#Convolutional Layer 5
model.add(Conv2D(base_hidden_units*4, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#Convolutional Layer 6
model.add(Conv2D(base_hidden_units*4, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Pool + Dropout
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# Fully Connected Layer 1
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.summary()


# %%
batch_size = 128
epochs = 40

#USe model checkpoint to save only the best model
checkpointer = ModelCheckpoint(filepath='model.weights.best.keras', verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')

#adam optimizer
opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# %%
import torch    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %%
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf



import tensorflow as tf
tf.config.list_physical_devices()


# %%
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=[checkpointer], verbose=2, shuffle=True)

# %%
scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
print("\nTest result: %.3f loss: %.3f" % (scores[1]*100,scores[0]))

# %%
# plot learning curves
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train-accuracy')
plt.plot(history.history['val_accuracy'], label='val-accuracy')
plt.legend()
plt.show()

#save this graph
plt.savefig('accuracy.png')

# %%
# Plot traning Loss and validation loss
plt.plot(history.history['loss'], label='train-loss')
plt.plot(history.history['val_loss'], label='val-loss')
plt.legend()
plt.show()

# save this graph
plt.savefig('loss.png')

# %%
#Save the history in a file named accuracies.txt (loss, everthing)
with open('accuracies.txt', 'w') as f:
    
    # Save Training Loss
    f.write('Training Loss: ')
    f.write(str(history.history['loss']))
    f.write('\n')
    
    # Save Validation Loss
    f.write('\nValidation Loss: ')
    f.write(str(history.history['val_loss']))
    f.write('\n')
    
    # Save Training Accuracy
    f.write('\nTraining Accuracy: ')
    f.write(str(history.history['accuracy']))
    f.write('\n')
    
    # Save Validation Accuracy
    f.write('\nValidation Accuracy: ')
    f.write(str(history.history['val_accuracy']))
    f.write('\n')
    
    

# %%


# %%


# %%


# %%



