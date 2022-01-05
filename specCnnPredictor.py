import os
import pickle

import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd

import sklearn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

from scipy.signal import spectrogram

import tensorflow as tf
import tensorflow.ragged
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.data import Dataset as tfDataset

import tensorflow_addons as tfa

import sys

xTrainCacheFile = "X_train.pkl"
yTrainCacheFile = "Y_train.pkl"

numFrequencies = 30 #number of frequencies in spectrogram

modelFilename = "specCnnModel"

defaultTestOuputFilename = "yTestPred.csv"

def getSpectrogram(signal, normalize=False):
    nperseg = 400
    if normalize:
        signal = signal/np.abs(signal).max()
    return spectrogram(signal, fs=300, nperseg=nperseg, scaling='spectrum', noverlap=int(nperseg*0.64), mode='magnitude')[2].transpose() #note: mode can be psd (choose either psd or magnitude), magnitude, angle, phase or complex

def normalizeMatrixInPlace(mat):
    mat -= np.mean(mat.flatten())
    mat /= np.max(np.abs(mat).flatten())
    return mat


def plotSpectrogram(signal):
    spec = getSpectrogram(signal[300:], normalize=True)[:40]
    plt.figure(figsize=(10, 20))
    plt.imshow(spec.transpose())
    plt.show()
    plt.plot(signal)
    plt.show()

def plotSpecWithInfo(sigIdx):
    print(f"sample {sigIdx} of class {Y_train[sigIdx]}")
    plotSpectrogram(X_train[sigIdx])

def tfTrainValSplit(X, y, valSize=0.2, shuffle=True):
    trainIndices, valIndices = train_test_split(np.arange(len(y)), test_size=valSize, shuffle=shuffle)
    return tf.gather(X, trainIndices), tf.gather(y, trainIndices), tf.gather(X, valIndices), tf.gather(y, valIndices)

def plotConfusionMatrix(yPred, yTrue, numClasses=4):
    confMat = confusion_matrix(yTrue, yPred, labels=np.arange(numClasses, dtype=np.int32))
    confMatDf = pd.DataFrame(confMat, index=np.arange(numClasses, dtype=np.int32), columns=np.arange(numClasses, dtype=np.int32))
    plt.figure()
    ax = sn.heatmap(confMatDf, annot=True)
    ax.set_xlabel("Expected class")
    ax.set_ylabel("Predicted class")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] == "--train":
        trainLabels = pd.read_csv('y_train.csv', index_col='id')
        trainData = pd.read_csv('X_train.csv', index_col='id')
        
        X_train = np.array([row.dropna().to_numpy() for i, row in trainData.iterrows()])
        Y_train = trainLabels.values.ravel()
        
        
       
        # trainDs = tfDataset.from_tensor_slices(X_train, Y_train)
        # trainDs = trainDs.map(lambda x, y: (getSpectrogram(x[300:], normalize=True)[:numFrequencies], y))
        # trainDs = trainDs.cache()
        
        trainSpecs = tf.expand_dims(tf.ragged.constant([getSpectrogram(x[300:], normalize=True)[:, :numFrequencies] for x in X_train]), axis=-1)
        yTrain_oneHot = tf.one_hot(Y_train, 4)
        
        classWeights = dict(enumerate(compute_class_weight("balanced", classes=np.unique(Y_train), y=Y_train)))
        
        def createModel(dropout=0.5):
            model = keras.Sequential()
            model.add(layers.Input(shape=(None, numFrequencies, 1), ragged=False))
            initialNumChannels = 16
            model.add(layers.Conv2D(initialNumChannels, (5,5), activation="relu", padding="same"))
            model.add(layers.MaxPool2D())
            #model.add(layers.Dropout(dropout))
            model.add(layers.Conv2D(2*initialNumChannels, (5,5), activation="relu", padding="same"))
            model.add(layers.MaxPool2D())
            #model.add(layers.Dropout(dropout))
            model.add(layers.Conv2D(4*initialNumChannels, (3,3), activation="relu", padding="same"))
            model.add(layers.MaxPool2D())
            numFeaturesPerTimestep = model.output_shape[-2]*model.output_shape[-1]
            model.add(layers.Reshape((-1, numFeaturesPerTimestep)))
            #model.add(layers.GlobalMaxPooling1D()) #should pool in penultimate dimension
            model.add(layers.GlobalAvgPool1D())
            model.add(layers.Dropout(dropout))
            model.add(layers.Flatten())
            #model.add(layers.Dense(numFeaturesPerTimestep//2))
            #model.add(layers.Dropout(dropout))
            model.add(layers.Dense(4, activation="softmax"))
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=4e-4), loss="categorical_crossentropy", metrics=['accuracy', tfa.metrics.F1Score(num_classes=4, average="micro")])
            model.summary()
            return model
        
        
        #xTrain, yTrain, xVal, yVal = train_test_split(trainSpecs, yTrain_oneHot, test_size=0.2, shuffle=True)
        xTrain, yTrain, xVal, yVal = tfTrainValSplit(trainSpecs, yTrain_oneHot, valSize=0.2, shuffle=True)
        
        trainDs = tf.data.Dataset.from_tensor_slices((xTrain, yTrain)).map(lambda x, y: (x.to_tensor(), y)).batch(1)
        valDs = tf.data.Dataset.from_tensor_slices((xVal, yVal)).map(lambda x, y: (x.to_tensor(), y)).batch(1)
        
        
        minTimeDimension = 1000
        for x, y in trainDs:
            #print(f"X type: {type(x)}, X shape: {x.shape}, y type: {type(y)}, y shape: {y.shape}")
            if x.shape[1] < minTimeDimension:
                minTimeDimension = x.shape[1]
        print(f"minTimeDimension: {minTimeDimension}")
        
        
        model = createModel()
        #history = model.fit(trainSpecs, yTrain_oneHot, batch_size=1, epochs=50, validation_split=0.2, shuffle=True, class_weight=classWeights)
        #history = model.fit(xTrain, yTrain, batch_size=1, epochs=50, validation_data=(xVal, yVal), shuffle=True, class_weight=classWeights)
        earlyStoppingCallback = EarlyStopping(monitor='val_f1_score', mode='max', patience=10, restore_best_weights=True)
        history = model.fit(trainDs, batch_size=1, epochs=120, validation_data=valDs, shuffle=True, class_weight=classWeights, callbacks=[earlyStoppingCallback])
        
        print(history.history)
        
        model.save(modelFilename)
        
        
        
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.set_title("Loss")
        ax1.plot(history.history['loss'], color='blue', label='train loss')
        ax1.plot(history.history['val_loss'], color='orange', label='val loss')
        ax1.legend()
        ax2.set_title("F1 Score")
        ax2.plot([np.mean(f1Score) for f1Score in history.history['f1_score']], color='blue', label='train F1 score')
        ax2.plot([np.mean(f1Score) for f1Score in history.history['val_f1_score']], color='orange', label='val F1 score')
        ax2.legend()
        plt.show()
        
        
        
        
        yValPred = model.predict(valDs.map(lambda x, y: x), batch_size=1)
        yValPred = tf.math.argmax(yValPred, axis=-1)
        plotConfusionMatrix(yValPred, tf.math.argmax(yVal, axis=-1))
    elif sys.argv[1] == '--test':
        print("running in prediction mode")
        model = keras.models.load_model(modelFilename)
        testData = pd.read_csv('X_test.csv', index_col='id')
        X_test = np.array([row.dropna().to_numpy() for i, row in testData.iterrows()])

        testSpecs = tf.expand_dims(tf.ragged.constant([getSpectrogram(x[300:], normalize=True)[:, :numFrequencies] for x in X_test]), axis=-1)

        testDs = tf.data.Dataset.from_tensor_slices((testSpecs,)).map(lambda x: x.to_tensor()).batch(1).cache()

        outputFilename = defaultTestOuputFilename if len(sys.argv) < 3 else sys.argv[2]

        pred = tf.math.argmax(model.predict(testDs, batch_size=1), axis=-1)
        outDf = pd.DataFrame({'y': pred})
        outDf.to_csv(outputFilename, index_label='id')
    else:
        print(f"error: unsupported arguments: {sys.argv[1:]}", file=sys.stderr)
