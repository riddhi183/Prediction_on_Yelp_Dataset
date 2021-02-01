#importing libraries
import json
import pandas as pd
import numpy as np
from numpy import savetxt
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

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


if len(sys.argv) < 3:
  print('===========================================================')
  print("Usage - ./run.sh preprocess-text-2 <INPUT DIR> <OUTPUT DIR>")
  print('=========================Output===============================')
  print("<OUTPUT DIR>/<OUTPUT FILE>")
  exit()


input_file = sys.argv[1]
output_path = sys.argv[2]


nltk.download('stopwords')

nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')

#loading transformed json data collected.json
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
df.head()

#dataframe as no blanks values
df.isnull().sum()

df['Ratings'].hist()

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)

train = pd.get_dummies(train, columns = ['Ratings'])
test = pd.get_dummies(test, columns = ['Ratings'])

class_names = ['Ratings_1', 'Ratings_2', 'Ratings_3', 'Ratings_4', 'Ratings_5']
y = train[class_names].values

"""#### Download it from here
#### https://nlp.stanford.edu/projects/glove/
"""

embed_size = 50 
# max number of unique words 
max_features = 2000
# max number of words from review to use
maxlen = 50
embedding_file = "Models/glove.6B.50d.txt"


def coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(coefs(*f.strip().split()) for f in open(embedding_file,mode="r", encoding="utf-8"))
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train['Reviews'].values))

X_train = tokenizer.texts_to_sequences(train['Reviews'].values)
X_test = tokenizer.texts_to_sequences(test['Reviews'].values)

x_train = pad_sequences(X_train, maxlen = maxlen)
x_test = pad_sequences(X_test, maxlen = maxlen)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():
    if i >= max_features:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

savetxt(output_path, embedding_matrix, delimiter=',')