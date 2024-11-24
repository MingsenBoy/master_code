import transformers
import torch
import random
import os
# new_working_directory = r'D:\Lab\project\paper\BioAug-main\script'
# os.chdir(new_working_directory)
import pandas as pd
import sys
import copy
from utils import get_random_gauss_value, linearize, mask_spacy_entities, add_relations, merge_list
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import json
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from pathlib import Path
# In[]
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--model', help='which model to use')
parser.add_argument('--input_file','-i', help='input file to use')
parser.add_argument('--sample_generation_mode',  default='dynamic', help='static/dynamic generation')
parser.add_argument('--directory','-dir', default='attn', help='data directory where train and dev files are located')
parser.add_argument('--topk', default='50', help='topk value')
parser.add_argument('--num_of_sequences', type=int, default=2, help='number of sequences per datapoint')
parser.add_argument('--max_length', default=150, help='max_length of the generated sentence')
parser.add_argument('--do_sample', default='True', help='do_sample argument')
parser.add_argument('--num_beams', default=5, help='num_beams argument')
# parser.add_argument('--file_name','-f', type=str, default='', help='file name for output')
# parser.add_argument('--root_dir','-ro', type=str, default='', help='root directory')
parser.add_argument('--seed', '-s', type=int, default=-1, help='random seed')
parser.add_argument('--mean', '-mean', type=float, default=0.7, help='mean for gauss prob')
parser.add_argument('--std', '-std', type=float, default=0.1, help='std_dev for gauss prob')
parser.add_argument('--shouldLinearizeAllWords', type=int, default=1, help='linearize mode')
parser.add_argument('--cuda_device', type=int, default=0, help='GPU number')
parser.add_argument('--template', type=str, help='dataset template')
parser.add_argument('--output_file', help='output file name')
args = parser.parse_args()
print(args)

# args.model = r"D:\Lab\project\paper\bioaug4re\datasets-pre\TACRED\train_processed-5-3-42--tokenfix-final"
# args.input_file = "train_processed"
# args.sample_generation_mode = "dynamic"
# args.directory = r"D:\Lab\project\paper\bioaug4re\datasets-pre\TACRED"
# args.topk = 10
# args.num_of_sequences = 2
# parser.add_argument('--max_length', default=150, help='max_length of the generated sentence')
# In[]
def add_relation_mentions(sketch, original_text, id, id2, verbalized_relation, full_train):
    if 'train' in id:
        full_data = full_train
        
    verbalized_relation_id_list = [data['id'] for data in verbalized_relation]
    id_index = id.split("-")[1]
    target_data_id = full_data[int(id_index)]['id']
    verbalized_relation_id = verbalized_relation_id_list.index(target_data_id)
    ver_relation = verbalized_relation[verbalized_relation_id][data_col]
        
    temp = []
    temp.append(f'</s> {ver_relation}')
    
    # id_index = id2.split("-")[1]
    target_data_id = full_data[int(id2)]['id']
    verbalized_relation_id = verbalized_relation_id_list.index(target_data_id)
    ver_relation = verbalized_relation[verbalized_relation_id][data_col]
    temp.append(f'</s> {ver_relation}')
    
    sketch += temp
    sketch = merge_list(sketch)
    sketch = sketch.replace(' </s> ', '</s>')
    return sketch

def remove_tags(temp):
    return ' '.join([i for i in temp.split() if i[0] != '<'])

# sent = (generated_text[0]['generated_text'])
def isGeneratedSentenceValid(sent):
    global new_tokens

    #sent = [element for element in sent if element != '']
    count = 0
    for i in sent.split(' '):
        if i != '':
            if (i[0] == '<' and i[-1] != '>') or (i[0] != '<' and i[-1] == '>'):
                return False
                #print("F")

            if i[0] == '<' and i[-1] == '>':
                if not i in new_tokens:
                    #print("F2")
                    return False
                count += 1
    if count % 2:
        #print(count)
        return False
    
    label_b = sum(1 for item in sent.split(' ') if "<b" in item)
    label_i = sum(1 for item in sent.split(' ') if "<i" in item)
    #label_b = sum(1 for item in sent if "<b" in item)
    if label_b == 0 or label_i % 2 != 0 or label_b % 2 != 0:
        return False
    
    return True

# entity = entities[2], origin_list = target_list, new_list = target_list2, between_indices = between_indices2
def search_idx(entity, origin_list, new_list, between_indices):
    
    target_word = origin_list[entity['start']]
    entity['start'] = new_list.index(target_word)
    while True:
        if entity['start'] not in between_indices:
            entity['start'] = new_list.index(target_word, entity['start'] + 1)
        else:
            break
    ent_start = entity['start']
    
    target_word2 = origin_list[entity['end'] - 1]
    entity['end'] = new_list.index(target_word2)
    while True:
        if entity['end'] not in between_indices:
            entity['end'] = new_list.index(target_word2, entity['end'] + 1)
        else:
            entity['end'] = entity['end'] + 1
            break
    ent_end = entity['end']
    
    return ent_start, ent_end
# In[]

if "TACRED" in args.directory:
    data_col = "verbalize_result"
elif "RE-TACRED" in args.directory:
    data_col = "relation_mention"

# cuda_device = 7
torch.cuda.set_device(args.cuda_device)

if not args.seed == -1:
    transformers.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
model_path = Path(os.path.join(args.directory,args.model))
model_path = model_path.resolve()

tokenizer = AutoTokenizer.from_pretrained(model_path)
genius = pipeline("text2text-generation", model=model_path,
                  tokenizer=tokenizer, device=args.cuda_device)

if args.directory[-1] != '/':
    args.directory += '/'

# newFileName = r'D:\Lab\project\paper\BioAug-main\datasets-precompute\Re-TACRED\3_dis\train_processed.json'
# newFileName = args.directory + args.input_file + '.json'
with open(os.path.join(args.directory,args.input_file + '.json'), 'r') as f:
    data = json.load(f)

    
# file_path2 = r"D:\Lab\project\paper\BioAug-main\datasets-precompute\Re-TACRED\3_dis\full_train_processed.json"
# file_path2 = f"{args.directory}full_train_processed.json"
with open(os.path.join(args.directory,'full_train_processed.json'), 'r') as f:
    full_train = json.load(f)

relation_list = []
origin_id_list = []
obj_type_list = []
subj_type_list = []
new_id_list = []

for row in full_train:
    relation_list.append(row['relation'])
    origin_id_list.append(row['origin_id'])
    obj_type_list.append(row['obj_type'])
    subj_type_list.append(row['subj_type'])
    new_id_list.append(row['origin_id'])
  
# file_path3 = r"D:\Lab\project\paper\BioAug-main\datasets-precompute\Re-TACRED\3_dis\llm_verbalize_realtion.json"
with open(os.path.join(args.directory,args.template + '.json'), 'r') as f:
    llm_verbalize_realtion = json.load(f)

text = [i['sentence'] for i in data]
types = [i['type'] for i in data]
labels = [i['labels'] for i in data]
id = [i['id'] for i in data]

model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
embeddings = model.encode(text, show_progress_bar=True, batch_size=1024, device=args.cuda_device)

text_similar = []
for i in range(len(embeddings)):
    arr = []
    for j in range(len(embeddings)):
        arr.append(1 - cosine(embeddings[i], embeddings[j]))

    top_2 = sorted(range(len(arr)), key=lambda i: arr[i])[-3:-1]
    if i in top_2:
        top_2 = sorted(range(len(arr)), key=lambda i: arr[i])[-4:-1]
        top_2.remove(i)
    # top_2 = sorted(range(len(arr)), key=lambda k: arr[k] if k != i else float('-inf'))[-3:-1]
    text_similar.append(top_2)

del model, embeddings

new_tokens = ['<b-country>', '<i-country>', '<b-religion>', '<i-religion>', '<b-url>', '<i-url>', '<b-person>', '<i-person>',
              '<b-location>', '<i-location>', '<b-ideology>', '<i-ideology>', '<b-organization>', '<i-organization>',
              '<b-criminal_charge>', '<i-criminal_charge>', '<b-cause_of_death>', '<i-cause_of_death>',
              '<b-misc>', '<i-misc>', '<b-city>', '<i-city>', '<b-state_or_province>', '<i-state_or_province>',
              '<b-date>', '<i-date>', '<b-title>', '<i-title>', '<b-nationality>', '<i-nationality>',
              '<b-number>', '<i-number>', '<b-duration>', '<i-duration>']

mapping_list = []
for token in new_tokens:
    fix_token = token[3:-1]
    mapping_list.append(fix_token.upper())
mapping_list = list(set(mapping_list))

new_tokens = set(new_tokens)

mapping_list = [
    'STATE_OR_PROVINCE', 'IDEOLOGY', 'CAUSE_OF_DEATH', 'NATIONALITY', 'CITY', 'LOCATION', 'DATE', 
    'ORGANIZATION', 'MISC', 'COUNTRY', 'DURATION', 'TITLE', 'URL', 'PERSON', 'RELIGION', 'NUMBER', 'CRIMINAL_CHARGE']

data_list = []
valid_list = []
specific_list = []
# DYNAMIC MASKING NEW CODE
if args.sample_generation_mode == 'dynamic':
    #with open(generated_file, 'w') as the_file:
        test = 0
        for i in tqdm(range(len(text))):
            # break
            break_flag = False
            saved = {}
            # break
            for z in range(int(args.num_of_sequences)):
                # break
                new_text, new_type, new_label = copy.deepcopy(
                    text[i]), copy.deepcopy(types[i]), copy.deepcopy(labels[i])
                
                text_similar_index = text_similar[i][z]

                new_text += text[text_similar_index]
                new_type += types[text_similar_index]
                new_label += labels[text_similar_index]
                
                origin_entities = [
                    full_train[i]['token'][full_train[i]['subj_start']:(full_train[i]['subj_end'] + 1)], 
                    full_train[i]['token'][full_train[i]['obj_start']:(full_train[i]['obj_end'] + 1)],
                    full_train[text_similar_index]['token'][full_train[text_similar_index]['subj_start']:(full_train[text_similar_index]['subj_end'] + 1)],
                    full_train[text_similar_index]['token'][full_train[text_similar_index]['obj_start']:(full_train[text_similar_index]['obj_end'] + 1)],
                    ]
                         
                assert len(new_text) == len(new_type) == len(new_label)
                # print('Text: ', new_text)
                original_text = ' '.join(copy.deepcopy(new_text))
                linearize(new_text, new_label, args.shouldLinearizeAllWords)
                mask_spacy_entities(new_text, new_type, args.mean, args.std)
                generated_sketch = add_relation_mentions(new_text, original_text, id[i], text_similar_index, llm_verbalize_realtion, full_train)# 一句選五筆，
                # generated_sketch = merge_list(new_text)
                
                
                generated_text = genius(generated_sketch, num_beams=int(args.num_beams), top_k=int(
                    args.topk), do_sample=args.do_sample, max_length=int(args.max_length))
                # print('Generated: ', generated_text[0]['generated_text'], '\n\n')
                # continue
                # valid_list.append(generated_text)
                        
                if not isGeneratedSentenceValid(generated_text[0]['generated_text']):
                    test += 1
                    continue
                
                target_list = generated_text[0]['generated_text'].split(' ')
                target_list = [element for element in target_list if element != '']
                indices = [x for x, element in enumerate(target_list) if '<' in element]
                pairs = [indices[q:q+2] for q in range(0, len(indices), 2)]
                
                #篩選前後不同BIO-label、長度錯誤
                for pair in pairs:
                    if target_list[pair[0]] != target_list[pair[1]]:
                        break_flag = True
                        break
                    if pair[1] - pair[0] > 2:
                        break_flag = True
                        break
                if break_flag:
                    #print("True")
                    continue
                
                
                valid_list.append(target_list)
                tokens = []
                relations = []
                entities = []
                count = 0
                count_2 = 0
                indices = [i for i, element in enumerate(target_list) if '<' in element]
                pairs = [indices[i:i+2] for i in range(0, len(indices), 2)]
                b_indices = [i for i, bidx in enumerate(target_list) if '<b' in bidx]
                b_pairs = [b_indices[i:i+2] for i in range(0, len(b_indices), 2)]
                #normal_token = [element for element in target_list_m if '<' not in element]
                continue_flag = False
                
                for pair in b_pairs:
                    if pair[1] - pair[0] != 2:
                        continue_flag = True
                
                if continue_flag:
                    continue
                
                # 篩選i-label起始資料
                for b_idx in range(len(b_pairs)):
                    if b_idx ==  0:
                        invalid_condition = any(element.startswith('<i') for element in target_list[:b_pairs[b_idx][1]])
                        if invalid_condition:
                            continue_flag = True
                            #print("1")
                            break
                
                if indices[-1] + 1 == len(target_list):
                    continue_flag = True
                    
                if continue_flag:
                    #count_ = count_ + 1
                    #print(valid_list.index(target_list))
                    continue
                         
                for j in range(len(indices)):
                    #break
                    if target_list[indices[j]][0:2] == '<b' and count == 0:
                        start = indices[j]
                        count = 1 
                        #pos = indices[i]
                    
                    elif count == 1:
                        if target_list[indices[j]][0:2] == '<i' and count_2 == 0:
                            count_2 = 1
                        elif target_list[indices[j]][0:2] == '<i' and count_2 == 1:
                            count_2 = 0
                            if target_list[indices[j] + 1][0:1] != '<' or target_list[indices[j] + 1][0:2] == '<b':
                                match_type = [item for item in mapping_list if item.upper() == target_list[indices[j]][3:-1].upper()]
                                ent_type = match_type[0]
                                end = indices[j]
                                ent_dict = {'type': ent_type, 'start': start, 'end': end}
                                entities.append(ent_dict)
                                count = 0
                                #print("2")
                        elif target_list[indices[j] + 1][0:2] == '<b' or target_list[indices[j] + 1][0:1] != '<':
                            #print("test")
                            match_type = [item for item in mapping_list if item.upper() == target_list[indices[j]][3:-1].upper()]
                            ent_type = match_type[0]
                            end = indices[j]
                            ent_dict = {'type': ent_type, 'start': start, 'end': end}
                            entities.append(ent_dict)
                            count = 0
                            #print("1")
                
                ent_value = []
                for ent in entities:
                    ent_value.append(ent['start'])
                    ent_value.append(ent['end'])
                
                if max(indices) != max(ent_value):
                    continue_flag = True
                
                # if len(entities) != 4:
                if len(entities) < 2:
                    specific_list.append(target_list)
                    continue_flag = True
                    
                if continue_flag:
                    continue
                            
                between_indices = [int((p[1]+p[0])/2) for p in pairs]
                            
                for ent in entities:
                    # break
                    if ent['end'] - ent['start'] > 1:
                        target = [ele for ele in target_list[ent['start']:ent['end']] if '<' not in ele]
                        head = target[0]
                        tail = target[-1]
                        head_pos = target_list.index(head)
                        tail_pos = target_list.index(tail)
                        while True:
                            if head_pos not in between_indices:
                                head_pos = target_list.index(head, head_pos + 1)
                            else:
                                break
                        while True:
                            if tail_pos not in between_indices:
                                tail_pos = target_list.index(tail, tail_pos + 1)
                            else:
                                break
                        
                        ent['start'] = head_pos
                        ent['end'] = tail_pos + 1
                    else:
                        continue_flag = True
                        break
                        
                if continue_flag:
                    continue

                while True:                     
                    new_id = ''.join(random.sample(origin_id_list[i], len(origin_id_list[i])))  
                    if new_id not in new_id_list:
                        new_id_list.append(new_id)
                        break

                mixner_entities = []
                
                for ent in entities:
                    # break
                    m_entity = [tar for tar in target_list[ent['start']:ent['end']] if "<" not in tar]
                    mixner_entities.append(m_entity)
                
                new_data = {
                    "origin_id": new_id, "relation": relation_list[i], "token": target_list,
                    "subj_start": full_train[i]['subj_start'], "subj_end": full_train[i]['subj_end'], "obj_start": full_train[i]['obj_start'], "obj_end": full_train[i]['obj_end'],
                    "subj_type": full_train[i]['subj_type'], "obj_type": full_train[i]['obj_type'], 'entities': entities, 
                    "old_id": origin_id_list[i], "old_id2": origin_id_list[text_similar_index], 
                    "origin_entities": origin_entities, "mixner_entities": mixner_entities
                    }

                data_list.append(new_data) 
# In[]       
output_file = f"{args.directory}/{args.output_file}.json"

with open(output_file, 'w') as f:
    json.dump(data_list, f)