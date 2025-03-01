📰 Fake News Detection using LSTM

📌 Project Overview

This project implements a Fake News Detection System using a Long Short-Term Memory (LSTM) network. The model analyzes news article text and classifies it as Fake or True with 99% accuracy.

📂 Dataset

The dataset consists of two files:

Fake.csv - Contains 23,502 fake news articles.

True.csv - Contains 21,417 true news articles.

Dataset Columns:

Title - Title of the news article.

Text - Full news content.

Subject - Category of the news.

Date - Publish date.

🏗 Model Architecture

The model is implemented using Keras and TensorFlow:

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.layers import BatchNormalization

embedding_size = 100

model = Sequential()
model.add(Embedding(vocab_sz + 1, embedding_size, mask_zero=True, input_length=maxlen))
model.add(LSTM(100, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))
model.add(Dropout(0.3))
model.add(LSTM(100, recurrent_dropout=0.2, dropout=0.2))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

🛠 Preprocessing Steps

Text Cleaning - Removing punctuation, special characters, and converting to lowercase.

Tokenization - Converting text into sequences using Tokenizer.

Padding - Ensuring uniform input length with pad_sequences().

Splitting Dataset - Dividing data into train (80%) and test (20%) sets.

📊 Model Performance

Test Accuracy: 99%

Evaluation Metrics: Precision, Recall, F1-score.

Confusion Matrix: Used to analyze classification errors.

🖥 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/hagagfawzi/Fake-News-Detection-using-LSTM.git
cd Fake-News-Detection-using-LSTM

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Model

python train.py

🚀 Next Steps & Improvements

✅ Implement Bidirectional LSTM (BiLSTM) for better context understanding.✅ Integrate Pretrained Embeddings (GloVe, Word2Vec) for richer representations.✅ Deploy as a Web API for real-time Fake News detection.

📜 License

This project is open-source and available under the MIT License.
