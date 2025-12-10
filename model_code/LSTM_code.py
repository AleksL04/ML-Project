from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def initLSTM():
    # Input 1: Text Data (Sequences)
    input_text = Input(shape=(50, 100), name='text_input')
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(input_text)
    x = GlobalMaxPooling1D()(x)

    # Input 2: Manual Features (Scalars)
    input_feat = Input(shape=(2,), name='feat_input')

    # Merge: Concatenate the processed text and the manual features
    combined = Concatenate()([x, input_feat])

    # Dense Layers
    z = Dense(16, activation='relu')(combined)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[input_text, input_feat], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def trainLSTM(X_train, y_train, X_valid, y_valid):
    # X_train is expected to be a list: [X_train_text, X_train_feat]
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