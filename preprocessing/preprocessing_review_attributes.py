import json
import csv
import math
import sys
import numpy as np
import pandas as pd


if len(sys.argv) < 4:
  print('===========================================================')
  print("Usage - ./run.sh preprocess-text-attr <INPUT ATTRIBUTE CSV> <INPUT REVIEW JSON> <OUTPUT DIR>")
  print('=========================Output===============================')
  print("<OUTPUT DIR>/merged.csv")
  print("<OUTPUT DIR>/review.csv")
  exit()

attribute_file = sys.argv[1]
review_file = sys.argv[2]
output_dir = sys.argv[3]

features = pd.read_csv(attribute_file)
labels = features['stars'].apply(lambda x: math.floor(x) )
features.drop(columns=["stars","Unnamed: 0"], inplace=True)
features = pd.concat([features,labels], axis=1)
print(features.columns)

"""Converting review.json to a DataFrame
Table joins to merge attribute data, rating data and review data into a single Df.
"""

df_review = []
with open(review_file,encoding='utf-8') as f:
  df_review,count = [],0
  lines = f.readlines()
  for line in lines:
    count+=1
    df_review.append(json.loads(line))


#Writting to CSV incase program crashes
f = open(output_dir+'/review.csv','w')
csv_file = csv.writer(f)
count = 0
for item in df_review:
  if count == 0:
    header = item.keys()
    csv_file.writerow(header)
    count += 1
  csv_file.writerow(item.values())  # ← changed
f.close()

del(df_review)

#Reading and merging from csv
df_review = pd.read_csv(output_dir+'/review.csv', chunksize=1000000)
for chunk in df_review:
  chunk.drop(columns=['review_id', 'user_id','useful', 'funny','cool','date','stars'], inplace=True)
  data = features.merge(chunk,on="business_id")

#Final File for multimodal classification with Text + Attributes
data.to_csv(output_dir+'/merged.csv')
