from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

def initLSTM():
    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(16, activation='relu')) # Intermediate dense layer
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def trainLSTM(X_train, y_train, X_valid, y_valid):
    model = initLSTM()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=20,
        batch_size=32,
        verbose=0,
        callbacks=callbacks
    )
    return model

def train_CV_LSTM(X_train, y_train, X_valid, y_valid):
    model = initLSTM()

    model = KerasClassifier(model=model, epochs=10, batch_size=32, verbose=0)
