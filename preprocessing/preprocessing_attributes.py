import json
import sys
import pandas as pd
from pandas.io.json import json_normalize
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA


pd.options.display.max_columns = None

if len(sys.argv) < 6:
  print('===========================================================')
  print("Usage - ./run.sh preprocess-attributes <INPUT DIR> <OUTPUT DIR> <OUTPUT FILE> <LIMIT > 100> <PCA COUNT <= 500>")
  print('=========================Output===============================')
  print("<OUTPUT DIR>/collected_baseline.json")
  print("<OUTPUT DIR>/<OUTPUT FILE>")
  exit()


dataset_path = sys.argv[1]
output_path = sys.argv[2]
output_file = sys.argv[3]
limit = int(sys.argv[4])
if limit < 100 and limit != -1:
  print("Limit > 100 or Limit -1")
  exit()
pca_num = int(sys.argv[5])
if pca_num > 500:
  print("Maximum 500 principal components can be generated")
  exit()

print("Loading data from "+dataset_path)

def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except ValueError as e:
    return False
  return True


def attempt_json_conv(s):
  if type(s) == str:
    if '{' in s and '}' in s:
      s = s.split(': ')
      s = ": '".join(s)
      s = s.split(', ')
      s = "', ".join(s)
      s = s.split('}')
      s = "'}".join(s)
    try:
      s = eval(s)
      return s
    except:
      return False



def flatten_json(nested_json):
    out = {}
    def flatten(x, name=''):
        s = attempt_json_conv(copy.deepcopy(x))
        if type(s) is dict:
          x = s
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            if type(x) == str:
              if x.lower() == 'TRUE'.lower():
                x = 1
              elif x.lower() == 'FALSE'.lower():
                x = 0
              elif x.lower() == 'nan'.lower() or x.lower() == 'None'.lower() or x.lower() == 'NA'.lower() or x == "{}":
                x = '0'
            out[name[:-1]] = x
    flatten(nested_json)
    return out


def strip_u(s):
  s = str(s).strip()
  if len(s) > 0:
    if s.lower() == 'nan'.lower() or s.lower() == 'None'.lower() or s.lower() == 'NA'.lower():
      s = ""
    else:
      if s[0] == 'u':
        s = s[1:len(s)]
      if s[0] == "'":
        s = s[1:len(s)]
      if s[len(s)-1] == "'":
        s = s[0:len(s)-1]
  return s


def convert_to_time(s):
  s = str(s).strip()
  if s == '0:0' or s == '0:00' or s == '00:00':
    return '2400'
  else:
    if len(s) > 0:
      s = s.split(':')
      if len(s[1]) == 1:
        s[1] += '0'
      s = ''.join(s)
    else:
      s = '0'
    return s


final_data = []

i = 0
with open(dataset_path+'/business.json') as file:
    for jsonObj in file:
        j = json.loads(jsonObj)
        data = flatten_json(j)
        final_data.append(data)
        i += 1
        if limit != -1 and i == limit:
          break


with open(output_path+'/collected_baseline.json', 'w') as file:
    json.dump(final_data,file)

final_data = pd.DataFrame.from_dict(final_data)

#Preprocessing and cleaning data
categories = final_data['categories'].str.get_dummies(sep=',').add_prefix('categories_')
attributes_RestaurantsAttire = final_data['attributes_RestaurantsAttire'].apply(lambda x: strip_u(x)).str.get_dummies().add_prefix('attributes_RestaurantsAttire_')
attributes_WiFi = final_data['attributes_WiFi'].apply(lambda x: strip_u(x)).str.get_dummies().add_prefix('attributes_WiFi_')
attributes_Alcohol = final_data['attributes_Alcohol'].apply(lambda x: strip_u(x)).str.get_dummies().add_prefix('attributes_Alcohol_')
attributes_NoiseLevel = final_data['attributes_NoiseLevel'].apply(lambda x: strip_u(x)).str.get_dummies().add_prefix('attributes_NoiseLevel_')
attributes_BYOBCorkage = final_data['attributes_BYOBCorkage'].apply(lambda x: strip_u(x)).str.get_dummies().add_prefix('attributes_BYOBCorkage_')
attributes_Smoking = final_data['attributes_Smoking'].apply(lambda x: strip_u(x)).str.get_dummies().add_prefix('attributes_Smoking_')
attributes_AgesAllowed = final_data['attributes_AgesAllowed'].apply(lambda x: strip_u(x)).str.get_dummies().add_prefix('attributes_AgesAllowed_')



final_data.drop(columns=['hours' ,'name', 'categories', 'attributes_RestaurantsAttire', 'attributes_WiFi', 'attributes_Alcohol', 'attributes_NoiseLevel', 'attributes_BYOBCorkage', 'attributes_AgesAllowed', 'attributes_Smoking', 'address', 'city', 'state',	'postal_code'],inplace=True)
final_data = pd.concat([final_data, categories, attributes_RestaurantsAttire, attributes_WiFi, attributes_Alcohol, attributes_NoiseLevel, attributes_BYOBCorkage, attributes_Smoking, attributes_AgesAllowed], axis=1)


#Preprocessing the open hours
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
for day in days:
  final_data[['hours_'+day+'_start','hours_'+day+'_close']] = final_data['hours_'+day].str.split('-',expand=True)
  hours_start = final_data['hours_'+day+'_start'].apply(lambda x: strip_u(x)).apply(lambda x: convert_to_time(x))
  hours_close = final_data['hours_'+day+'_close'].apply(lambda x: strip_u(x)).apply(lambda x: convert_to_time(x))
  final_data.drop(columns=['hours_'+day,'hours_'+day+'_start','hours_'+day+'_close'], inplace=True)
  final_data = pd.concat([final_data, hours_start, hours_close], axis=1)

final_data = final_data.fillna(0)

#Dropping Attribute Columns
final_columns = [col for col in final_data.columns if col.startswith('attributes_')]
final_data.drop(columns=final_columns, inplace=True)

#Generating PCA
labels = final_data['stars'].apply(lambda x: str(x))
business_ids = final_data['business_id'].apply(lambda x: str(x))
final_data.drop(columns=['stars','business_id'], inplace=True)
features = final_data
features = StandardScaler().fit_transform(features)

pca = PCA(n_components=pca_num)
prinComps = pca.fit_transform(features)
columns = []
for i in range(pca_num):
  columns.append('pca_'+str(i))

features = pd.DataFrame(data=prinComps, columns=columns)
features = pd.concat([business_ids,features,labels],axis=1)

features.to_csv(output_path+'/'+output_file)
