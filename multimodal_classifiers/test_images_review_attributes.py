import tensorflow as tf
from tensorflow.python.keras.layers import Multiply,multiply
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.preprocessing.text import tokenizer_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from random import sample

import json
import cv2
import glob
import pickle
import math
import numpy as np
import pandas as pd
import os
import sys


if len(sys.argv) < 3:
  print('===========================================================')
  print("Usage - ./run.sh test-text-img-attr <INPUT DIR> <MERGED ATTRIBUTE FILE> <TESTING SIZE Default=0.10>")
  print('=========================Output===============================')
  print("Testing Accuracy")
  exit()

input_dir = sys.argv[1]
attribute_file = sys.argv[2]
bs = 0.10
if len(sys.argv) == 4:
  bs = float(sys.argv[3])

#Retrieving merged Review and Attributes
data = pd.read_csv(attribute_file)
print(data.columns)

#Converting photo.json to a DataFrame
with open(input_dir+'/photo.json',encoding='utf-8') as f:
   photos = []
   lines = f.readlines()
   for line in lines:
     photos.append(json.loads(line))
   
df = pd.DataFrame(photos)
df = df[df["label"] == "food"]

data = df.merge(data,on="business_id")
X1,X2,X3,Y = data["photo_id"],data["text"],data.drop(columns=["business_id","photo_id","stars","label","caption","text","Unnamed: 0", data.columns[0]]),data["stars"]

del(df)

with open('Models/tokenizer.pickle', 'rb') as t:
    tokenizer = pickle.load(t)

X2 = X2.apply(lambda x: str(x))

X2 = tokenizer.texts_to_sequences(X2)
X2 = pad_sequences(X2, maxlen = 685)

X3 = X3.to_numpy()
Y = Y.to_numpy()

#Defining the multimodal model
attributes,text,img = load_model("Models/attribute_model.h5"),load_model("Models/txt_model.h5"),load_model("Models/img_model.h5")
fuse = multiply([attributes.output,text.output,img.output])
MM_model = Model(inputs=[attributes.input,text.input,img.input],outputs=[fuse])

bs = int(math.floor(float(data.shape[0]) * bs))

entries = os.listdir(input_dir+'/photos')

#x1_data,x2_data,y_data = np.empty([bs,224,224,3]),np.empty([bs,500]),np.empty([bs,5])
batch = sample(list(zip(X1,X2,X3,Y)),bs)

x1_2 = []
x2_2 = []
x3_2 = []
y_2 = []

for j in range(len(batch)):
  x1,x2,x3,y = batch[j]
  if x1+'.jpg' not in entries:
    continue
  #print(x1)
  img = cv2.imread(input_dir+"/photos/{0}.jpg".format(x1),1)
  img = cv2.resize(img,(224,224))
  x1_2.append(img/255)
  x2_2.append(x2)
  x3_2.append(x3)
  y_2.append(y)

x1_data = np.array(x1_2)
x2_data = np.array(x2_2)
x3_data = np.array(x3_2)
y_data = np.array(y_2)


y_pred = MM_model.predict([x3_data,x2_data,x1_data])
y_actual = y_data

count,correct = 0,0
for i,j in zip(y_pred,y_actual):
   print(i,j)
   if(np.argmax(i) == j):
      correct +=1 
   count+=1

print("Predicted Samples ",count)
print("accuracy",correct/count)

