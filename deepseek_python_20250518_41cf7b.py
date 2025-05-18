import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class FakeNewsDetector:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.vectorizer = None
        
    def clean_text(self, text):
        """
        Clean and preprocess text data
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def load_data(self, filepath):
        """
        Load and preprocess dataset
        """
        df = pd.read_csv(filepath)
        
        # Clean text data
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Map labels to binary (0 for real, 1 for fake)
        df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})
        
        return df
    
    def train_model(self, df):
        """
        Train the fake news detection model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label'], test_size=0.2, random_state=42
        )
        
        # Create pipeline with TF-IDF and PassiveAggressiveClassifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_df=0.7)),
            ('pac', PassiveAggressiveClassifier(max_iter=50))
        ])
        
        # Hyperparameter tuning
        parameters = {
            'tfidf__max_df': (0.5, 0.75, 1.0),
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'pac__C': [0.01, 0.1, 1.0]
        }
        
        grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.vectorizer = grid_search.best_estimator_.named_steps['tfidf']
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
    def predict(self, text):
        """
        Predict whether a news article is fake or real
        """
        if not self.model:
            raise Exception("Model not trained. Please train the model first.")
            
        cleaned_text = self.clean_text(text)
        prediction = self.model.predict([cleaned_text])
        
        return "Fake News" if prediction[0] == 1 else "Real News"
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        """
        import joblib
        joblib.dump((self.model, self.vectorizer), filepath)
        
    def load_saved_model(self, filepath):
        """
        Load a pre-trained model from disk
        """
        import joblib
        self.model, self.vectorizer = joblib.load(filepath)


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Load dataset (example path - replace with your dataset)
    # Dataset format should have 'text' and 'label' columns
    try:
        df = detector.load_data("fake_news_dataset.csv")
        print("Dataset loaded successfully")
        
        # Train model
        detector.train_model(df)
        
        # Save model
        detector.save_model("fake_news_detector.pkl")
        
        # Test prediction
        test_news = """
        Scientists have discovered a new planet that could support human life.
        The planet, named Kepler-452b, is located 1,400 light-years away.
        """
        print("\nTest News:", test_news)
        print("Prediction:", detector.predict(test_news))
        
    except FileNotFoundError:
        print("Dataset file not found. Please provide the correct path to your dataset.")