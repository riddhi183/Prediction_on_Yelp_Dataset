from tensorflow.python.keras.layers import Multiply,multiply
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.preprocessing.text import tokenizer_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from random import sample

import json
import cv2
import glob
import pickle
import sys
import math
import os
import numpy as np
import pandas as pd 

if len(sys.argv) < 2:
  print('===========================================================')
  print("Usage - ./run.sh test-image-text <INPUT DIR> <TESTING SIZE Default=0.10>")
  print('=========================Output===============================')
  print("Testing Accuracy")
  exit()


input_dir = sys.argv[1]
bs = 0.10
if len(sys.argv) == 3:
  bs = float(sys.argv[2])

#Converting photo.json to a DataFrame
with open(input_dir+'/photo.json',encoding='utf-8') as f:
   data = []
   lines = f.readlines()
   for line in lines:
     data.append(json.loads(line))
   
df = pd.DataFrame(data)
df = df[df["label"] == "food"]


#Converting business.json to a DataFrame
with open(input_dir+'/business.json',encoding='utf-8') as f2:
   data_score = []
   lines = f2.readlines()
   for line in lines:
     data_score.append(json.loads(line))
   
df_score = pd.DataFrame(data_score)

#Converting review.json to a DataFrame
with open(input_dir+'/review.json',encoding='utf-8') as f:
   data_review,count = [],0
   lines = f.readlines()
   for line in lines:
     count+=1
     data_review.append(json.loads(line))
     if count ==1000:
        break


#Table joins to merge photo data, rating data and review data into a single Df.
df_review = pd.DataFrame(data_review)
data = df.merge(df_score[["business_id","stars"]],on="business_id")
data = data.merge(df_review[["business_id","text"]],on="business_id")

X1,X2,Y = data["photo_id"],data["text"],data["stars"] 


#Binning the Output Classes
for i in range(len(Y)):
   Y[i] = int(Y[i]-1)


# Loading the Tokenizer
with open('Models/tokenizer.pickle', 'rb') as t:
    tokenizer = pickle.load(t)

X2 = tokenizer.texts_to_sequences(X2)
X2 = pad_sequences(X2, maxlen = 685)


#Defining the multimodal model
txt,img = load_model("Models/txt_model.h5"),load_model("Models/img_model.h5")
fuse = multiply([txt.output,img.output])
MM_model = Model(inputs=[txt.input,img.input],outputs=[fuse])


#Creating a batch for testing - sampling without replacement
bs = int(math.floor(float(data.shape[0]) * bs))
print(bs)

entries = os.listdir(input_dir+'/photos/')

#x1_data,x2_data,y_data = np.empty([bs,224,224,3]),np.empty([bs,685]),np.empty([bs,5])

x1_2 = []
x2_2 = []
y_2 = []

batch = sample(list(zip(X1,X2,Y)),bs)
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

#Making Predictions using the model
y_pred = MM_model.predict([x2_data,x1_data])
y_actual = y_data


# Manually calculating accuracy as the model needs to be compiled for model.evaluate.
count,correct = 0,0
for i,j in zip(y_pred,y_actual):
   print(i,j)
   if(np.argmax(i) == j):
      correct +=1 
   count+=1

print("Predicted Samples ",count)
print("accuracy",correct/count)




