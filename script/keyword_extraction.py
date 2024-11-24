import scispacy
import spacy
# In[]
import os
import json 
import pandas as pd
import copy
import re
import string
import random
import math
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--sample_size', default=3, help='data sample size 0.03/0.1/0.15/all')
parser.add_argument('--data_type', default="train", help='train or dev')
parser.add_argument('--file_path', help='where train and dev data are located')
parser.add_argument('--output_path', help='path to the outputs')

args = parser.parse_args()

if args.sample_size == "all":
    args.sample_size = 1
else:
    args.sample_size = round(int(args.sample_size)/100, 2)
# args.sample_size = 0.03
# args.data_type = "train"
# args.file_path = r"D:\Lab\project\paper\bioaug4re\dataset\TACRED"
# args.output_path = r"D:\Lab\project\paper\bioaug4re\datasets-pre\TACRED"
# In[]
def is_punctuation(element):
    return element in string.punctuation
# In[]
if args.data_type == "train":
    file = "train.json"
elif args.data_type == "dev":
    file = "dev.json"

with open(os.path.join(args.file_path,file), 'r') as f:
    data_list = json.load(f)

with open(os.path.join(args.file_path,"relation_dic.json"), 'r') as f:
    relation_dic = json.load(f) 

random_sample_data = []
# random_sample_data = data_list2
# In[]
# 分層抽樣
sample_percent = args.sample_size
distratefied_sample_data = []
for rel in relation_dic:
    # break
    # if rel == "NA":
    #     rel = "no_relation"
    # proportion = relation_dic[rel]/len(data_list)
    # sample_percent = 0.15
    temp_data_list = []
    temp_data_list = list(filter(lambda x: x['relation']== rel, data_list))
    sample_size = math.ceil(len(temp_data_list) * sample_percent) 
    if sample_size < 1:
        sample_size = 1
    print(rel + ": " + str(sample_size))
    
    distratefied_sample_data = distratefied_sample_data + random.sample(temp_data_list, sample_size)
# In[]
complete_data_list = []
nlp = spacy.load("en_core_sci_scibert")
count = 0
#m_list = []

for data in tqdm(distratefied_sample_data, desc="Processing Data"):
    token_list = [token for token in data['token']]
    relation = data['relation']
    docid = data["docid"]
    data_id = data["id"]
    # print(count)
    id = "{}-{}".format(args.data_type, count)
    count = count + 1
    
    ent_list = []
    # 主、副實體放入list
    if data['subj_start'] < data['obj_start']:
        
        ent_dic = {
            'ent_type': data['subj_type'], 'start': data['subj_start'], 'end': data['subj_end']
            }
        ent_list.append(ent_dic)
        ent_dic2 = {
            'ent_type': data['obj_type'], 'start': data['obj_start'], 'end': data['obj_end']
            }
        ent_list.append(ent_dic2)
    else:
        
        ent_dic = {
            'ent_type': data['obj_type'], 'start': data['obj_start'], 'end': data['obj_end']
            }
        ent_list.append(ent_dic)
        ent_dic2 = {
            'ent_type': data['subj_type'], 'start': data['subj_start'], 'end': data['subj_end']
            }
        ent_list.append(ent_dic2)
    
    # 原始label(未合併) 加入BIO標籤
    origin_labels = ['O']*len(token_list)
    for label in ent_list:
        #break
        label_start = label['start']
        label_end = label['end']
        if (label_end - label_start) >= 1:
            for k in range(label_start, label_end+1):
                if k == label_start:
                    origin_labels[k] = "<B-{}>".format(label['ent_type'])
                else:
                    origin_labels[k] = "<I-{}>".format(label['ent_type'])
        else:
            origin_labels[label_start] = "<B-{}>".format(label['ent_type'])
    
    #BIO labels合併
    for label in ent_list:
        # if sen_label.index(element) == 0:
        #     break
        #origin_labels_start = origin_labels[(element[1])]
        if ent_list.index(label) == 0:
            #print("pass")
            if label['start'] == 0: # 起點開始之實體
                #print("pass")
                new_labels = []
                #labels = []
                combine_label = ' '.join(origin_labels[(label['start']):(label['end'] + 1)])
                new_labels = new_labels + [combine_label]
                origin_labels_start = label['end'] + 1
            elif int(label['start'] - 1) == 0: 
                #print("pass")
                new_labels = ['O']*1
                combine_label = ' '.join(origin_labels[(label['start']):(label['end'] + 1)])
                new_labels = new_labels + [combine_label]
                origin_labels_start = label['end'] + 1
            else:
                #print("pass")
                new_labels = ['O']*int(label['start'])
                combine_label = ' '.join(origin_labels[(label['start']):(label['end'] + 1)])
                new_labels = new_labels + [combine_label]
                origin_labels_start = label['end'] + 1
            
        elif label['start'] == origin_labels_start:
            #print("pass")
            combine_label = ' '.join(origin_labels[(label['start']):(label['end'] + 1)])
            new_labels = new_labels + [combine_label]
            origin_labels_start = label['end'] + 1
        else:
            #print("pass")
            mid_label = ['O']*int((label['start'] - origin_labels_start))
            combine_label = ' '.join(origin_labels[(label['start']):(label['end'] +1)])
            new_labels = new_labels + mid_label + [combine_label]
            origin_labels_start = label['end'] + 1
            
        # 兩實體標註完後剩餘labels
        if ent_list.index(label) == len(ent_list) - 1 :
            # print("pass")
            final_label = ['O']*(len(origin_labels) - origin_labels_start)
            new_labels = new_labels + final_label
    
    ##BIO標籤tokens合併
    for sen in ent_list:
        # if sen_label.index(sen) == 1:
        #     break
        #origin_labels_start = origin_labels[(element[1])]
        if ent_list.index(sen) == 0:
            #print("pass")
            if sen['start'] == 0: # 起點開始之實體
                #print("pass")
                new_token_list = []
                combine_sen = ' '.join(token_list[(sen['start']):(sen['end'] + 1)])
                new_token_list = new_token_list + [combine_sen]
                last_end = sen['end'] + 1
            elif int(sen['start'] - 1) == 0: 
                #print("pass")
                new_token_list = [token_list[0]]
                combine_sen = ' '.join(token_list[(sen['start']):(sen['end'] + 1)])
                new_token_list = new_token_list + [combine_sen]
                last_end = sen['end'] + 1
            else:
                #print("pass")
                new_token_list = token_list[0:sen['start']]
                combine_sen = ' '.join(token_list[(sen['start']):(sen['end'] + 1)])
                new_token_list = new_token_list + [combine_sen]
                last_end = sen['end'] + 1
            
        elif sen['start'] == last_end:
            #print("pass")
            combine_sen = ' '.join(token_list[(sen['start']):(sen['end'] + 1)])
            new_token_list = new_token_list + [combine_sen]
            last_end = sen['end'] + 1
        else:
            #print("pass")
            mid_sen = token_list[last_end:sen['start']]
            combine_sen = ' '.join(token_list[(sen['start']):(sen['end'] + 1)])
            new_token_list = new_token_list + mid_sen + [combine_sen]
            last_end = sen['end'] + 1
        
        # 兩實體標註完後剩餘tokens
        if ent_list.index(sen) == len(ent_list) - 1 :
            #print("pass")
            final_sen = token_list[last_end:len(token_list)]
            new_token_list = new_token_list + final_sen 
            
    type_list = new_labels
    
    labels = []
    # 刪除"<>"
    for i in range(len(new_labels)):
        if new_labels[i] != 'O':
            #break
            modify_label = new_labels[i].replace("<", "").replace(">", "")    
            labels.append(modify_label)
        else:
            labels.append(new_labels[i])
    #m_list.append(labels)
    
    # NE標籤list
    for l in range(len(type_list)):
            if type_list[l] != 'O':
                #print("pass")
                type_list[l] = 'NE'
    # token合併成string            
    for t in range(len(token_list)):
        if t == 0:
            combined_text = token_list[t]
            continue
        
        if is_punctuation(token_list[t]):
            combined_text = "".join([combined_text, token_list[t]])
        else:
            combined_text = " ".join([combined_text, token_list[t]])
            
    doc = nlp(combined_text) 
    
    # 實體標籤list
    for ent in doc.ents:
        if str(ent) in new_token_list:
            list_index = new_token_list.index(str(ent))
            if type_list[list_index] != 'NE':
                type_list[list_index] = 'E'
            #break
    
    new_data = {"origin_id": data_id, "relation": relation, "token": data['token'], 
                "subj_start": data["subj_start"], "subj_end": data["subj_end"],
                "obj_start": data["obj_start"], "obj_end": data["obj_end"],
                "subj_type": data["subj_type"], "obj_type": data["obj_type"],
                "id": id, "labels": labels, "sentence": new_token_list, "type": type_list}
                 
    complete_data_list.append(new_data)
    # complete_data_list = complete_data_list + data_list2
# In[]
da_data_list = []  
count = 0        
for row in complete_data_list:
    id = "{}-{}".format(args.data_type, count)
    new_data = {"id": id, 'labels': row['labels'], 'sentence': row['sentence'], "type": row['type']}
    # break
    count += 1
    da_data_list.append(new_data)
    

with open(os.path.join(args.output_path,"train_processed.json"), 'w') as file:
    json.dump(da_data_list, file)
# In[]
# count = 0
for data in complete_data_list:
    data['id'] = data['origin_id']
    # count += 1

with open(os.path.join(args.output_path,"full_train_processed.json"), 'w') as file:
    json.dump(complete_data_list, file)
