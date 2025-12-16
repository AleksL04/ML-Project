from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def trainXGB(train_df, valid_df):
    # 1. Increase Vocabulary to 30,000 to catch names like "Grassley", "Edison"
    # 2. Use ngram_range=(1, 3) to capture phrases like "melts into oblivion"
    vectorizer = TfidfVectorizer(
        max_features=30000,       
        ngram_range=(1, 3),        
        analyzer='word',
        stop_words='english'
    )
    
    # XGBoost is excellent at handling these sparse features
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    pipe = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', model)
    ])
    
    print("Training XGBoost with expanded vocabulary...")
    pipe.fit(train_df['text'], train_df['label'])
    
    # Validation
    preds = pipe.predict(valid_df['text'])
    acc = accuracy_score(valid_df['label'], preds)
    print(f"XGBoost Validation Accuracy: {acc:.4f}")
    
    return pipe

def predictXGB(model, df):
    # Since model is a Pipeline, it handles vectorization automatically
    preds = model.predict(df['text'])
    # Get probability of Class 1 (Satire) for the Blender
    probas = model.predict_proba(df['text'])[:, 1]
    return preds, probas