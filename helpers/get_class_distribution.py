import random
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt


if len(sys.argv) < 3:
  print('===========================================================')
  print("Usage - ./run.sh get-class-dist <INPUT CSV FILE> <OUTPUT DIR>")
  print('=========================Output===============================')
  print("<OUTPUT DIR>/class_distribution.png")
  exit()

input_file = sys.argv[1]
output_path = sys.argv[2]

features = pd.read_csv(input_file)
labels = np.array(features['stars'].apply(lambda x: str(x)))

classes = np.unique(labels)
print("The unique labels are "+str(classes))

y_values = labels
unique, counts = np.unique(y_values, return_counts=True)
y_train = y_values[:80]
y_valid = y_values[81:]

plt.bar(unique, counts, 1)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
#plt.show()

plt.savefig(output_path+'/class_distribution.png')