
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from src.preprocess import preprocess_data, load_data

def train_model():
    df = load_data()
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = SVC()
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'model.joblib')
    
    # Evaluate
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    return accuracy

if __name__ == "__main__":
    train_model()


