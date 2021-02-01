import tensorflow as tf
from tensorflow.python.keras.layers import Multiply,multiply
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.preprocessing.text import tokenizer_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from random import sample

import json
import cv2
import glob
import math
import numpy as np
import pandas as pd
import os
import sys

if len(sys.argv) < 3:
  print('===========================================================')
  print("Usage - ./run.sh test-image-attr <INPUT DIR> <ATTRIBUTE FILE> <TESTING SIZE Default=0.10>")
  print('=========================Output===============================')
  print("Testing Accuracy")
  exit()


input_dir = sys.argv[1]
attribute_file = sys.argv[2]
bs = 0.10
if len(sys.argv) == 4:
  bs = float(sys.argv[3])


features = pd.read_csv(attribute_file)
labels = features['stars'].apply(lambda x: math.floor(x) )
features.drop(columns=["stars","Unnamed: 0"], inplace=True)
features = pd.concat([features,labels], axis=1)

#Converting photo.json to a DataFrame
with open(input_dir+'/photo.json',encoding='utf-8') as f:
   data = []
   lines = f.readlines()
   for line in lines:
     data.append(json.loads(line))
   
df = pd.DataFrame(data)
df = df[df["label"] == "food"]

data = df.merge(features,on="business_id")
X1,X2,Y = data["photo_id"],data.drop(columns=["business_id","photo_id","stars","label", data.columns[0]]),data["stars"] 
print(X2.columns)
#Binning the Output Classes
for i in range(len(Y)):
   Y[i] = int(Y[i]-1)

X2 = X2.to_numpy()
Y = Y.to_numpy()

#Defining the multimodal model
attributes,img = load_model("Models/attribute_model.h5"),load_model("Models/img_model.h5")
fuse = multiply([attributes.output,img.output])
MM_model = Model(inputs=[attributes.input,img.input],outputs=[fuse])

bs = int(math.floor(float(data.shape[0]) * bs))

entries = os.listdir(input_dir+'/photos/')

batch = sample(list(zip(X1,X2,Y)),bs)

x1_2 = []
x2_2 = []
y_2 = []

for j in range(len(batch)):
  x1,x2,y = batch[j]
  if x1+'.jpg' not in entries:
    continue
  img = cv2.imread(input_dir+"/photos/{0}.jpg".format(x1),1)
  img = cv2.resize(img,(224,224))
  x1_2.append(img/255)
  x2_2.append(x2)
  y_2.append(y)

x1_data = np.asarray(x1_2)
x2_data = np.asarray(x2_2)
y_data = np.asarray(y_2)


y_pred = MM_model.predict([x2_data,x1_data])
y_actual = y_data

count,correct = 0,0
for i,j in zip(y_pred,y_actual):
   print(i,j)
   if(np.argmax(i) == j):
      correct +=1 
   count+=1

print("Predicted Samples ",count)
print("accuracy",correct/count)

