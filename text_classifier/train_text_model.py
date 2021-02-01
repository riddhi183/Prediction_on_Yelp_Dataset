import json
import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt

import re
import string
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, GRU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers.embeddings import Embedding


if len(sys.argv) < 2:
  print('===========================================================')
  print("Usage - ./run.sh train-text-model <INPUT JSON>")
  print('=========================Output===============================')
  print("Training Accuracy")
  exit()


input_file = sys.argv[1]

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

with open(input_file) as f:
    json_data = json.load(f)

def wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

#preparing list for converting the json data to a dataframe
text_list=[]
rating_list=[]
for i in json_data:
    for j in json_data[i]['reviews']:
        input_str = j['text']
        input_str = input_str.lower()
        input_str = [word.strip(string.punctuation) for word in input_str.split(" ")]
        input_str = [word for word in input_str if not any(c.isdigit() for c in word)]
        stop = stopwords.words('english')
        input_str = [x for x in input_str if x not in stop]
        input_str = [t for t in input_str if len(t) > 0]
        pos_tags = pos_tag(input_str)
        input_str = [WordNetLemmatizer().lemmatize(t[0], wordnet_pos(t[1])) for t in pos_tags]
        input_str = [t for t in input_str if len(t) > 1]
        input_str = " ".join(input_str)
        text_list.append(input_str)
        rating_val=j['stars']
        rating_list.append(int(rating_val))


#Dataframe with required columns
df = pd.DataFrame(list(zip(text_list, rating_list)), columns =['Reviews', 'Ratings']) 
df.shape

# to save preprocesseded data
df.to_csv(r'df.csv', index = False)

#upload the preprocessed data
df = pd.read_csv(r"df.csv")
df.head()
df.isnull().sum()
df = df.dropna()
df['Ratings'].hist()

#converting df to list 
docs_reviews=df['Reviews'].tolist()
labels=df['Ratings'].tolist()

#tokenizing the reviews
words_list = []
for i in docs_reviews:
    tokenized_word = word_tokenize(i)
    for j in tokenized_word:
        words_list.append(j)

#no of unique words
vocab_size = len(set(words_list))
print(vocab_size)

# to find max length of review
w_count = lambda sen: len(word_tokenize(sen))
lngst_sen = max(docs_reviews, key=w_count)
maxlen = len(word_tokenize(lngst_sen))


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2,random_state= 0)

train = pd.get_dummies(train, columns = ['Ratings'])
test = pd.get_dummies(test, columns = ['Ratings'])

class_names = ['Ratings_1', 'Ratings_2', 'Ratings_3', 'Ratings_4', 'Ratings_5']

y = train[class_names].values
y_test = test[class_names].values

embed_size = 50

embedding_file = "Models/glove.6B.50d.txt"

def coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(coefs(*f.strip().split()) for f in open(embedding_file,mode="r", encoding="utf-8"))
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(list(train['Reviews'].values))

X_train = tokenizer.texts_to_sequences(train['Reviews'].values)
X_test = tokenizer.texts_to_sequences(test['Reviews'].values)
x_train = pad_sequences(X_train, maxlen = maxlen)
x_test = pad_sequences(X_test, maxlen = maxlen)


embed_size=50
word_index = tokenizer.word_index
#print(word_index)
embedding_matrix = np.zeros((vocab_size, embed_size))

for word, i in word_index.items():
    if i >= vocab_size:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#build LSTM model 
model = Sequential()
model.add(Embedding(vocab_size, embed_size,trainable = False ,weights = [embedding_matrix], input_length=maxlen))
model.add(LSTM(24, input_shape=(1200, 19), return_sequences=True, implementation=2))
model.add(Flatten())
model.add(Dense(5, activation='sigmoid'))

#compile the model with poisson loss
model.compile(optimizer='adam', loss='poisson', metrics=['acc'])
model.fit(x_train, y, batch_size = 1200, epochs = 20, validation_split = .1)
model.save("Models/txt_model.h5")
y_pred = model.predict([x_test], batch_size=1024, verbose = 1)
accuracy =model.evaluate(x_test, y_test, verbose = 1, batch_size=100)

print('test loss, test acc:', accuracy)