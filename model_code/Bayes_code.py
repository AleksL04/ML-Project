from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, log_loss

def initBayes():
    return MultinomialNB()

def trainBayes(train_df, valid_df):

    train_texts = train_df["text"].values
    y_train = train_df["label"].values

    valid_texts = valid_df["text"].values
    y_valid = valid_df["label"].values
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_valid = vectorizer.transform(valid_texts)

    model = initBayes()
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds)

    probas = model.predict_proba(X_valid)
    loss = log_loss(y_valid, probas)

    print("Validation Accuracy:", acc)
    print("Validation Loss:", loss)

    return model, vectorizer

def predictBayes(model, df, vectorizer):

    text = df["text"].values
    y = df["label"].values

    text = vectorizer.transform(text)
    preds = model.predict(text)
    acc = accuracy_score(y, preds)
    probas = model.predict_proba(text)

    pred = probas[:,1]   # probability of sarcasm (class 1)

    return preds, acc, pred

