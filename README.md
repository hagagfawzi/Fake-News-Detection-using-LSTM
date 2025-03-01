📰 Fake News Detection using LSTM

📌 Project Overview

This project implements a Fake News Detection System using a Long Short-Term Memory (LSTM) network. The model analyzes news article text and classifies it as Fake or True with 99% accuracy.

📂 Dataset

The dataset consists of two files:

Fake.csv (23,502 fake news articles)

True.csv (21,417 true news articles)

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

📊 Model Performance

Test Accuracy: 99%

Evaluation Metrics: Precision, Recall, F1-score.

Confusion Matrix: Analyzed to assess classification errors.

🔧 Installation & Setup

Clone this repository:

git clone https://github.com/hagagfawzi/Fake-News-Detection-using-LSTM.git
cd Fake-News-Detection-using-LSTM

Install dependencies:

pip install -r requirements.txt

Run the model:

python train.py

🚀 Next Steps & Improvements

Use Bidirectional LSTM (BiLSTM) for better context understanding.

Integrate Pretrained Embeddings (GloVe, Word2Vec) for richer representations.

Deploy as a Web API for real-time Fake News detection.

📜 License

This project is open-source and available under the MIT License.
