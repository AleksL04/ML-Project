from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalMaxPooling1D, Conv1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def initCNN():
    # Input 1: Text Data
    input_text = Input(shape=(50, 100), name='text_input')
    y = Conv1D(filters=64, kernel_size=3, activation='relu')(input_text)
    y = GlobalMaxPooling1D()(y)

    # Input 2: Manual Features
    input_feat = Input(shape=(2,), name='feat_input')

    # Merge
    combined = Concatenate()([y, input_feat])

    # Dense Layers
    z = Dense(8, activation='relu')(combined)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[input_text, input_feat], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def trainCNN(X_train, y_train, X_valid, y_valid, sample_weights):
    # X_train is expected to be a list: [X_train_text, X_train_feat]
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