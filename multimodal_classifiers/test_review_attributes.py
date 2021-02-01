import tensorflow as tf
from tensorflow.python.keras.layers import Multiply,multiply
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.preprocessing.text import tokenizer_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from random import sample

import json
import csv
import math
import cv2
import glob
import pickle
import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
  print('===========================================================')
  print("Usage - ./run.sh test-text-attr <INPUT DIR> <TESTING SIZE Default=0.10>")
  print('=========================Output===============================')
  print("Testing Accuracy")
  exit()


input_dir = sys.argv[1]
bs = 0.10
if len(sys.argv) == 3:
  bs = float(sys.argv[2])

"""Getting the attributes and extracting features and labels"""
data = pd.read_csv(input_dir+'/merged.csv')
print(data.columns)
X1,X2,Y = data["text"],data.drop(columns=["business_id","Unnamed: 0","text","stars", data.columns[0]]),data['stars']


"""Binning the Output Classes"""
for i in range(len(Y)):
   Y[i] = int(Y[i]-1)

with open('Models/tokenizer.pickle', 'rb') as t:
    tokenizer = pickle.load(t)

X1 = X1.apply(lambda x: str(x))

X1 = tokenizer.texts_to_sequences(X1)
X1 = pad_sequences(X1, maxlen = 685)

X2 = X2.to_numpy()
Y = Y.to_numpy()


"""Creating a batch for testing - sampling without replacement"""
bs = int(math.floor(float(data.shape[0]) * bs))
x1_data,x2_data,y_data = np.empty([bs,685]),np.empty([bs,500]),np.empty([bs,5])


batch = sample(list(zip(X1,X2,Y)),bs)
for j in range(len(batch)):
   x1_data[j],x2_data[j],y_data[j] = batch[j]

attributes,review = load_model("Models/attribute_model.h5"),load_model("Models/txt_model.h5")
fuse = multiply([attributes.output,review.output])
MM_model = Model(inputs=[attributes.input,review.input],outputs=[fuse])

y_pred = MM_model.predict([x2_data,x1_data])
y_actual = y_data

count,correct = 0,0
for i,j in zip(y_pred,y_actual):
   print(i,j)
   if(np.argmax(i) == j[0]):
      correct +=1 
   count+=1
print("Predicted Samples ",count)
print("Accuracy",correct/count)