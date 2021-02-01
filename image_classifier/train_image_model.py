from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
#from vgg16_places_365 import VGG16_Places365 as VGG16
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.python.keras.models import Model,load_model
from random import sample
from collections import Counter


import json
import cv2
import glob
import numpy as np
import pandas as pd 
import sys


if len(sys.argv) < 2:
  print('===========================================================')
  print("Usage - ./run.sh train-image-model <INPUT DIR>")
  print('=========================Output===============================')
  print("Training and validation accuracy")
  print("Models/image_model.h5")
  exit()


dataset_path = sys.argv[1]


# Generating new branches
def batch_generator(bs,data):
   while(1):
     x_data,y_data = np.empty([bs,224,224,3]),np.empty([bs,5])
     batch = sample(data,bs)
     for j in range(len(batch)):
       x,y_data[j] = batch[j]
       print(x)
       img = cv2.imread(dataset_path+"/photos/{0}.jpg".format(x),1)
       img = cv2.resize(img,(224,224))
       x_data[j] = img/255

     yield x_data,y_data



#Converting photo.json to a DataFrame
with open(dataset_path+'/photo.json',encoding='utf-8') as f:
   data = []
   lines = f.readlines()
   for line in lines:
     data.append(json.loads(line))
   
df = pd.DataFrame(data)
df = df[df["label"] == "food"]

#Converting business.json to a DataFrame
with open(dataset_path+'/business.json',encoding='utf-8') as f2:
   data_score = []
   lines = f2.readlines()
   for line in lines:
     data_score.append(json.loads(line))


#Table Joins on the photo data and the stars data   
df_score = pd.DataFrame(data_score)

data = df.merge(df_score[["business_id","stars"]],on="business_id")
group_by_id = data.groupby("business_id")

X = group_by_id["photo_id"].apply(list).tolist()
Y = group_by_id["stars"].apply(list).apply(lambda data: data[0]).tolist()


#additional data preprocessing - One Hot encoding and stratified sampling
label_dict = {0:[1,0,0,0,0],1:[0,1,0,0,0],2:[0,0,1,0,0],3:[0,0,0,1,0],4:[0,0,0,0,1]}

raw_data = []
for x,y in zip(X,Y):
   for i in x:
      raw_data.append((i,label_dict[int(y-1)]))

data = []
sample_size = 1600  
counts = [0 for i in range(5)]
for i in raw_data:
  for label in range(5):
     if(i[1]==label_dict[label] and counts[label]<sample_size):
       data.append(i)
       counts[label] += 1
  

#Creating the Model by concatenating it with the pre-trained model
frozen_model = MobileNetV2(include_top=False, weights='imagenet')
x = frozen_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=frozen_model.input, outputs=predictions)


#Freezing the convolutional layers
for layer in frozen_model.layers:
    layer.trainable = False

#compile the model
model.compile(optimizer='Adam',
              loss='poisson',
              metrics=['accuracy'])


#Training the model along with hyperparameter definitions
epochs,bs = 30,100
x_data,y_data = np.empty([bs,224,224,3]),np.empty([bs,1])
model.fit_generator(batch_generator(bs,data),epochs=50,steps_per_epoch=int(len(data)/bs))
model.save("Models/img_model.h5")


#Creating a batch for testing - sampling without replacement
x_data,y_data = np.empty([bs,224,224,3]),np.empty([bs,5])
batch = sample(data,bs)
for j in range(len(batch)):
  x,y_data[j] = batch[j]
  img = cv2.imread(dataset_path+"/photos/{0}.jpg".format(x),1)
  img = cv2.resize(img,(224,224))
  x_data[j] = img/255


#Evaluating the model
new_model = load_model("Models/img_model.h5")
accuracy = new_model.evaluate(x_data,y_data)
print(accuracy)

