import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Download necessary NLTK packages
nltk.download('stopwords')
nltk.download('vader_lexicon')  # Download VADER lexicon for sentiment analysis

# 1. Load your dataset
df = pd.read_csv('tweets_dataset.csv')
print("Original columns:", df.columns.tolist())

# 2. Use VADER for initial sentiment labeling
sid = SentimentIntensityAnalyzer()

def get_sentiment_label(text):
    if isinstance(text, float) and np.isnan(text):
        return 'Neutral'  # Handle NaN values
    
    scores = sid.polarity_scores(str(text))
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply VADER to create sentiment labels
df['Sentiment'] = df['Content'].apply(get_sentiment_label)
print("Sentiment distribution from VADER:", df['Sentiment'].value_counts())

# 3. Text Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    # Handle NaN values
    if isinstance(text, float) and np.isnan(text):
        return ""
    
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase and tokenize
    words = text.lower().split()
    
    # Remove stopwords and stem
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

df['Cleaned_Content'] = df['Content'].apply(clean_text)

# 4. Encode sentiment labels
encoder = LabelEncoder()
df['Encoded_Sentiment'] = encoder.fit_transform(df['Sentiment'])

# Save original sentiment labels 
sentiment_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Sentiment mapping:", sentiment_mapping)

# 5. Split data
X = df['Cleaned_Content'].values
y = df['Encoded_Sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Find the maximum sequence length
max_len = max(len(x) for x in X_train_seq)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Check for class imbalance
print("Class distribution in training data:", np.bincount(y_train))

# 7. LSTM Model
def create_lstm_model(num_classes=1):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
    model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    # Adjust the output layer based on number of classes
    if num_classes == 2:  # Binary (e.g., Positive vs Negative)
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:  # Multi-class (e.g., Positive, Negative, Neutral)
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

# 8. BiLSTM Model
def create_bilstm_model(num_classes=1):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
    model.add(Bidirectional(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    # Adjust the output layer based on number of classes
    if num_classes == 2:  # Binary
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:  # Multi-class
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

# Determine number of classes
num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")

# 9. Train LSTM model
lstm_model = create_lstm_model(num_classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lstm_history = lstm_model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# 10. Train BiLSTM model
bilstm_model = create_bilstm_model(num_classes)
bilstm_history = bilstm_model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# 11. Evaluate models
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_pad, y_test)
bilstm_loss, bilstm_accuracy = bilstm_model.evaluate(X_test_pad, y_test)

print(f"LSTM Model - Loss: {lstm_loss}, Accuracy: {lstm_accuracy}")
print(f"BiLSTM Model - Loss: {bilstm_loss}, Accuracy: {bilstm_accuracy}")

# 12. Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['accuracy'], label='LSTM Train')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM Validation')
plt.plot(bilstm_history.history['accuracy'], label='BiLSTM Train')
plt.plot(bilstm_history.history['val_accuracy'], label='BiLSTM Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['loss'], label='LSTM Train')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation')
plt.plot(bilstm_history.history['loss'], label='BiLSTM Train')
plt.plot(bilstm_history.history['val_loss'], label='BiLSTM Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# 13. Predict sentiment for new tweets
def predict_sentiment(text, model):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_len)
    
    if num_classes == 2:  # Binary classification
        prediction = model.predict(padded)[0][0]
        sentiment_idx = 1 if prediction >= 0.5 else 0
        confidence = prediction if prediction >= 0.5 else 1 - prediction
    else:  # Multi-class classification
        prediction = model.predict(padded)[0]
        sentiment_idx = np.argmax(prediction)
        confidence = prediction[sentiment_idx]
    
    # Convert index back to sentiment label
    for label, idx in sentiment_mapping.items():
        if idx == sentiment_idx:
            sentiment = label
            break
    
    return sentiment, float(confidence)

# Test with sample tweets
sample_tweets = [
    "AI is going to revolutionize the job market in Malaysia!",
    "Worried about losing my job to AI in the coming years",
    "The impact of AI on Malaysian jobs remains to be seen"
]

print("\nLSTM Model Predictions:")
for tweet in sample_tweets:
    sentiment, confidence = predict_sentiment(tweet, lstm_model)
    print(f"Tweet: {tweet}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")

print("\nBiLSTM Model Predictions:")
for tweet in sample_tweets:
    sentiment, confidence = predict_sentiment(tweet, bilstm_model)
    print(f"Tweet: {tweet}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")

# 14. Save the models
lstm_model.save('lstm_sentiment_model.h5')
bilstm_model.save('bilstm_sentiment_model.h5')

# Save tokenizer and encoder
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the sentiment distribution to a CSV file
sentiment_distribution = df['Sentiment'].value_counts().reset_index()
sentiment_distribution.columns = ['Sentiment', 'Count']
sentiment_distribution.to_csv('sentiment_distribution.csv', index=False)
