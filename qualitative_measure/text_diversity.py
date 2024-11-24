import json

import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--input_file', help='data input')
args = parser.parse_args()
print(args)
# In[]
train_data = json.load(open(args.input_file, 'r'))
origin_data_id = list(set([data['origin_id'] for data in train_data if "origin_id" in data.keys()]))
origin_data = [data for data in train_data if data['id'] in origin_data_id]
train_data_compare = [data for data in train_data if data['id'] not in origin_data_id]
diver_list = []
for idx, data in enumerate(origin_data):
    # break
    target_data_list = [data2 for idx2, data2 in enumerate(train_data_compare) if data2['origin_id'] == data['id']]
    total_new_tokens_percentage = 0
    if len(target_data_list) == 0:
        continue
    for idx2, data2 in enumerate(target_data_list):
        # break
        new_tokens = [token for token in data2['token'] if data2 not in data['token']]
        new_tokens_percentage = len(new_tokens) / len(data['token']) if data['token'] else 0
        total_new_tokens_percentage += new_tokens_percentage
    average_new_tokens_percentage = total_new_tokens_percentage / len(target_data_list)
    diver_list.append(average_new_tokens_percentage)
diver = sum(diver_list)/len(diver_list) 
print("diversity:" + str(diver))