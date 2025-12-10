from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D

def initCNN():
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def trainCNN(X_train, y_train, X_valid, y_valid, sample_weights):
    model = initCNN()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-5)
    ]

    history = model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        validation_data=(X_valid, y_valid),
        epochs=20,
        batch_size=32,
        verbose=0,
        callbacks=callbacks
    )

    return model