import argparse
import transformers
import torch
import os
# new_working_directory = r'D:\Lab\project\paper\BioAug-main\script'
# os.chdir(new_working_directory)
import sys
import copy
from transformers import pipeline, AutoTokenizer
import json
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--input_file', help='nomral_inference output')
parser.add_argument('--directory', help='data directory where pre_data are located')
parser.add_argument('--output_file', help='output file name')
parser.add_argument('--range',default='normal_generation', help='all/normal_generation')
args = parser.parse_args()
print(args)
# In[]
def sen_pool_score_4_generation(embedding_list):
    text_similar = []
    total_average = 0
    for i in range(len(embedding_list)):
        arr = []
        for j in range(len(embedding_list)):
            arr.append(1 - cosine(embeddings[i], embeddings[j]))
         
        # if i != 0:
        average = round(sum(arr)/len(arr), 3)
        total_average = total_average + average
        text_similar.append((i,average))
        # text_similar.append(average)
        
    text_similar = sorted(text_similar, key = lambda x: x[1])
    del_index = text_similar[0][0]
        
    average_sim = round(total_average/len(text_similar), 3)
    
    return average_sim, del_index
# In[]
# generation_file_path = r"{}\{}.json".format(args.directory,args.input_file)
with open(os.path.join(args.directory,args.input_file + ".json"), 'r') as f:
    generation_data = json.load(f)

if args.range == "normal_generation":
    mixner_list = [data for data in generation_data if 'old_id2' in data]
    generation_normal_data = []
    for idx, data in enumerate(generation_data):
        # if 'old_id' in data:
        #     break
        # if 'old_id2' not in data and 'orgin_id' in data:
        if 'old_id2' not in data:
            generation_normal_data.append(copy.deepcopy(data))
elif args.range == "all":
    generation_normal_data = generation_data
# generation_data = copy.deepcopy(generation_data2)
  
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
# embeddings = model.encode(text, show_progress_bar=True, batch_size=1024)
   
old_id_list = []
for idx,data in enumerate(generation_normal_data):
    # if data['relation'] == "no_relation":
    # old_id_list.append(data['origin_id'])
    try:
        old_id_list.append(data['origin_id'])
    except:
        old_id_list.append(data['orgin_id'])
        data['origin_id'] = data['orgin_id']
old_id_list = list(set(old_id_list))

all_origin_id_list = []
for idx, old_id in enumerate(old_id_list):
    # break
    target_id_data_list = []
    target_id_data_list = [gene['id'] for gene in generation_normal_data if gene['origin_id'] == old_id]
    all_origin_id_list.append(target_id_data_list)
    

all_text_similar = []
final_generation_list = []
for idx, id_list in enumerate(all_origin_id_list):
    # break
    target_data_list = []
    old_id = old_id_list[idx]
    
    target_data_list = target_data_list + [data for data in generation_data if data['id'] in id_list and "labels" not in data]
    if target_data_list == []:
        target_data_list = target_data_list + [data for data in generation_data if data['id'] in id_list and "labels" in data]
        final_generation_list = final_generation_list + target_data_list
        continue
    
    text = []
    text = [(idx, data['token']) for idx,data in enumerate(target_data_list)]
    embeddings = model.encode(text, show_progress_bar=True, batch_size=1024)
    
    if len(embeddings) < 2:
        average_sim = 1
    else:
        average_sim, del_index = sen_pool_score_4_generation(embeddings)
        del target_data_list[del_index] 
        
    # del target_data_list[del_index] 
    target_data_list = target_data_list + [data for data in generation_data if data['id'] in id_list and "labels" in data]
    
    final_generation_list = final_generation_list + target_data_list
    # average_sim = sen_pool_score(embeddings)
    all_text_similar.append(average_sim)

final_generation_list = final_generation_list + mixner_list
# In[]
# output_file_path = r"{}\{}_filter.json".format(args.directory, args.output_file)

with open(os.path.join(args.directory,args.output_file + ".json"), 'w') as f:
    json.dump(final_generation_list, f)