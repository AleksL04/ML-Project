from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def initBayes():
    model = GaussianNB()
    return model


def trainBayes(x_train, y_train, x_valid, y_valid):
    model = initBayes()
    model.fit(x_train, y_train)

    preds = model.predict(x_valid)
    acc = accuracy_score(y_valid, preds)

    print("Validation Accuracy:", acc)

    return model
