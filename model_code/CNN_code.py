from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import concatenate, Input
from tensorflow.keras.models import Model

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

import numpy as np

def initTextCNN(length, vocab_size):
    inputs = Input(shape=(length, vocab_size))
    
    # Parallel convolutions
    conv_blocks = []
    for kernel_size in [3, 4, 5]:
        conv = Conv1D(filters=128, kernel_size=kernel_size, activation='relu')(inputs)
        pool = GlobalMaxPooling1D()(conv)
        conv_blocks.append(pool)
    
    merged = concatenate(conv_blocks)
    dense = Dense(64, activation='relu')(merged)
    dropout = Dropout(0.5)(dense)
    outputs = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def initCNN():
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def trainCNN(X_train, y_train, X_valid, y_valid):
    #model = initCNN()
    model = initTextCNN()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-5)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=20,
        batch_size=32,
        verbose=0,
        callbacks=callbacks
    )

    return model

def train_CV_CNN(X_train, y_train, X_valid, y_valid):

    X = np.concatenate((X_train, X_valid), axis=0)

    y = np.concatenate((y_train, y_valid), axis=0)


    model = KerasClassifier(model=initCNN, epochs=10, batch_size=32, verbose=0)

    scores = cross_val_score(model, X, y, cv=5)

    print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

