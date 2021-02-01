import json
import sys

if len(sys.argv) < 3:
  print('===========================================================')
  print("Usage - ./run.sh preprocess-text <INPUT DIR> <OUTPUT DIR>")
  print('=========================Output===============================')
  print("<OUTPUT DIR>/collected.json")
  exit()

dataset_path = sys.argv[1]
output_path = sys.argv[2]



final_data= {}
user_count = 0
with open(dataset_path+'/user.json') as file:
  for jsonObj in file:
    j = json.loads(jsonObj)
    final_data[j['user_id']] = {}
    final_data[j['user_id']]['review_count'] = j['review_count']
    final_data[j['user_id']]['useful'] = j['useful']
    final_data[j['user_id']]['cool'] = j['cool']
    final_data[j['user_id']]['fans'] = j['fans']
    final_data[j['user_id']]['elite'] = j['elite']
    final_data[j['user_id']]['average_stars'] = j['average_stars']
    final_data[j['user_id']]['compliment_hot'] = j['compliment_hot']
    final_data[j['user_id']]['compliment_more'] = j['compliment_more']
    final_data[j['user_id']]['compliment_profile'] = j['compliment_profile']
    final_data[j['user_id']]['compliment_cute'] = j['compliment_cute']
    final_data[j['user_id']]['compliment_list'] = j['compliment_list']
    final_data[j['user_id']]['compliment_note'] = j['compliment_note']
    final_data[j['user_id']]['compliment_plain'] = j['compliment_plain']
    final_data[j['user_id']]['compliment_cool'] = j['compliment_cool']
    final_data[j['user_id']]['compliment_funny'] = j['compliment_funny']
    final_data[j['user_id']]['compliment_writer'] = j['compliment_writer']
    final_data[j['user_id']]['compliment_photos'] = j['compliment_photos']
    final_data[j['user_id']]['reviews'] = []
    final_data[j['user_id']]['tips'] = []
    user_count += 1


review_count = 0
with open(dataset_path+'/review.json') as file:
  for jsonObj in file:
    j = json.loads(jsonObj)
    n_r = {}
    n_r['business_id'] = j['business_id']
    n_r['stars'] = j['stars']
    n_r['text'] = j['text']
    n_r['useful'] = j['useful']
    n_r['funny'] = j['funny']
    n_r['cool'] = j['cool']
    final_data[j['user_id']]['reviews'].append(n_r)
    review_count += 1
    


tip_count = 0
with open(dataset_path+'/tip.json') as file:
  for jsonObj in file:
    j = json.loads(jsonObj)
    n_t = {}
    n_t['text'] = j['text']
    n_t['compliment_count'] = j['compliment_count']
    n_t['business_id'] = j['business_id']
    final_data[j['user_id']]['tips'].append(n_t)
    tip_count += 1
    
del(file)

#with open('/Users/prithvirajchaudhuri/Desktop/CSC522/CSC522-Project/Dataset/yelp_dataset/photo.json') as file:
# photo = json.load(file)

#with open('/Users/prithvirajchaudhuri/Desktop/CSC522/CSC522-Project/Dataset/yelp_dataset/checkin.json') as file:
#  checkin = json.load(file)

with open(output_path+'/collected.json', 'w') as file:
    json.dump(final_data,file)


print("Total Users Extracted "+str(user_count)+"\n")
print("Total Reviews Extracted "+str(review_count)+"\n")
print("Total Tips Extracted "+str(tip_count)+"\n")