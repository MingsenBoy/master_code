import json
import os

import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--input_file', help='mixner_inference output')
parser.add_argument('--input_file2', help='nomral_inference output')
parser.add_argument('--num_of_sequences', type=int, help='number of sequences per datapoint(mixner)')
parser.add_argument('--num_of_sequences2', type=int, default=2, help='number of sequences per datapoint(normal)')
# parser.add_argument('--dataset_directory', help='dataset directory where train and dev files are located')
parser.add_argument('--directory', help='data directory where pre_data are located')
parser.add_argument('--output_file', help='output file name')
args = parser.parse_args()
print(args)

full_train_name = "full_train_processed"
# outputname = f"model_train{args.num_of_sequences2}mix{args.num_of_sequences}_inference_ver"

# generation_file_path = r"{}\{}.json".format(args.directory, args.input_file)
with open(os.path.join(args.directory,args.input_file + ".json"), 'r') as f:
    generation = json.load(f)
    
# origin_file_path = r"{}\{}.json".format(args.directory, full_train_name)
with open(os.path.join(args.directory,full_train_name + ".json"), 'r') as f:
    origin_data = json.load(f)

old_list = []
for data in origin_data:
    old_list.append(data['id'])
# In[]
# 檢查前後idx是否差1 target_list = target_iidx_list
def i_position_2hop_check(target_list, data_flag):
    # data_flag = False
    for i in range(1,len(target_list),2):
        if i == len(target_list) - 1:
            continue
        elif target_list[i] + 1 != target_list[i + 1]:
            data_flag = True
            break
        
    return data_flag

# 檢查<i>符號位置是否正確
def check_i_position(b_pair_list, iidx_list, data_flag):
    
    for idx,b_pair in enumerate(b_pair_list):
        # break
        # print(idx)
        # 檢查0到第一個是否有出現<i>
        if idx == 0 and bool(set(list(range(0,b_pair[1]))) & set(iidx_list)):
            data_flag = True
            break
        # 檢查上一個<b>到下一個<b>的區間，檢查最後一個<b>到剩下之token
        elif idx == len(b_pair_list) - 1:
            target_idx_list = list(range(b_pair_list[idx-1][1] + 1, b_pair[0]))
            target_iidx_list = [value for value in iidx_list if value >= min(target_idx_list) and value <= max(target_idx_list)]
            data_flag = i_position_2hop_check(target_iidx_list, data_flag)
            
            target_idx_list = list(range(b_pair[1], len(data['token'])))
            # 如果最後一個<b>之後還有<i>符號
            if bool(set(target_idx_list) & set(iidx_list)):
                target_iidx_list = [value for value in iidx_list if value >= min(target_idx_list) and value <= max(target_idx_list)]
                data_flag = i_position_2hop_check(target_iidx_list, data_flag)
        # 檢查上一個<b>到下一個<b>的區間
        else:
             target_idx_list = list(range(b_pair_list[idx-1][1] + 1, b_pair[0]))
             if bool(set(target_idx_list) & set(iidx_list)):
                 target_iidx_list = [value for value in iidx_list if value >= min(target_idx_list) and value <= max(target_idx_list)]
                 data_flag = i_position_2hop_check(target_iidx_list, data_flag)
                 
    return data_flag
# In[] 
gene_final = []
# 篩掉起始位置end>start資料、重複實體資料
for idx,data in enumerate(generation):
    # breakx
    if len(data['entities']) > 0:
        length_flag = True
        #篩掉起始位置end>start資料
        for pos in data['entities']:
            if pos['end'] < pos['start']:
                length_flag = False
                #gene_final.remove(gene)  
                break
        #篩掉重複實體資料
        for i in range(len(data['entities'])):
            for j in range(i + 1, len(data['entities'])):
                if data['entities'][i] == data['entities'][j]:
                    length_flag = False
                    break
    if length_flag == False:
        #print("true")
        continue
            
    wrong_data_flag = False
    bidx_list = [t_index for t_index,token in enumerate(data['token']) if token[:2] == "<b"]
    iidx_list = [i_index for i_index,token in enumerate(data['token']) if token[:2] == "<i"]
    b_pair_list = [bidx_list[i:i+2] for i in range(0, len(bidx_list), 2)]
    i_pair_list = [iidx_list[i:i+2] for i in range(0, len(iidx_list), 2)]
    beetween_idx_list = [int((pair[1] + pair[0])/2) for pair in b_pair_list] + [int((pair[1] + pair[0])/2) for pair in i_pair_list]
    ent_type_list = [type['type'] for type in data['entities']]
    for idx,b_pair in enumerate(b_pair_list):
        # break
        # print(idx)
        # 檢查0到第一個是否有出現<i>
        if idx == 0 and bool(set(list(range(0,b_pair[1]))) & set(iidx_list)):
            wrong_data_flag = True
            break
        # 檢查上一個<b>到下一個<b>的區間，檢查最後一個<b>到剩下之token
        elif idx == len(b_pair_list) - 1:
            target_idx_list = list(range(b_pair_list[idx-1][1] + 1, b_pair[0]))
            if bool(set(target_idx_list) & set(iidx_list)):
                target_iidx_list = [value for value in iidx_list if value >= min(target_idx_list) and value <= max(target_idx_list)]
                wrong_data_flag = i_position_2hop_check(target_iidx_list, wrong_data_flag)
            
            target_idx_list = list(range(b_pair[1], len(data['token'])))
            # 如果最後一個<b>之後還有<i>符號
            if bool(set(target_idx_list) & set(iidx_list)):
                target_iidx_list = [value for value in iidx_list if value >= min(target_idx_list) and value <= max(target_idx_list)]
                wrong_data_flag = i_position_2hop_check(target_iidx_list, wrong_data_flag)
        # 檢查上一個<b>到下一個<b>的區間
        else:
             target_idx_list = list(range(b_pair_list[idx-1][1] + 1, b_pair[0]))
             if bool(set(target_idx_list) & set(iidx_list)):
                 target_iidx_list = [value for value in iidx_list if value >= min(target_idx_list) and value <= max(target_idx_list)]
                 wrong_data_flag = i_position_2hop_check(target_iidx_list, wrong_data_flag)
                   
    if wrong_data_flag:
        continue
        
    entities = []
    for idx,b_pair in enumerate(b_pair_list):
        # break
        # if (b_pair[1] + 1) not in iidx_list and (b_pair[1] + 1) not in bidx_list:
        if (b_pair[1] + 1) not in iidx_list:
            ent_data = {
                'type': ent_type_list[idx],
                'start': b_pair[0] + 1,
                'end': b_pair[1],
                }
            entities.append(ent_data)
        elif (b_pair[1] + 1) in iidx_list:
            for i in range(int(b_pair[1] + 1),len(data['token'])):
                if i not in bidx_list and i not in iidx_list and i not in beetween_idx_list:
                    end_pos = i - 1
                    ent_data = {
                        'type': ent_type_list[idx],
                        'start': b_pair[0] + 1,
                        'end': end_pos,
                        }
                    entities.append(ent_data)
                    break
        if idx == 1:
            if data['subj_start'] < data['obj_start']:
                data['subj_start'] = entities[0]['start']
                data['subj_end'] = entities[0]['end']
                data['obj_start'] = entities[1]['start']
                data['obj_end'] = entities[1]['end']
            else:
                data['obj_start'] = entities[0]['start']
                data['obj_end'] = entities[0]['end']
                data['subj_start'] = entities[1]['start']
                data['subj_end'] = entities[1]['end']
        
    data['entities'] = entities
    
    gene_final.append(data) 
       
   # In[] 
da_data_list = []
   
# 修改實體位置(與原生資料集相同包頭包尾)，token資料去除BIO標籤
for gene in gene_final:
    # break
    #del gene['tokens_m']
    #gene['tokens'] = gene['tokens_m']
    gene_m = [element for element in gene['token'] if '<' not in element] 
    for entity in gene['entities']:
        # break
        if entity['end'] - entity['start'] == 1:
            start = entity['start']   
            new_start = start - len([tag for tag in gene['token'][:start] if "<" in tag])
            
            entity['start'] = new_start
            entity['end'] = new_start
        else:
            start = entity['start']
            end = entity['end']
            
            new_start = start - len([tag for tag in gene['token'][:start] if "<" in tag])
            new_end = end - len([tag for tag in gene['token'][:end] if "<" in tag]) - 1
            
            entity['start'] = new_start
            entity['end'] = new_end
    gene['token'] = gene_m
    
    
# 更正subj_start~obj_end的實體位置
for idx, gene in enumerate(gene_final):
    # break
    old_id_idx = old_list.index(gene['old_id'])
    relation = origin_data[old_id_idx]['relation']
    gene['relation'] = relation
    
    if origin_data[old_id_idx]['subj_start'] < origin_data[old_id_idx]['obj_start']:
        gene['subj_start'] = gene['entities'][0]['start']
        gene['subj_end'] = gene['entities'][0]['end']
        gene['subj_type'] = gene['entities'][0]['type']
        
        gene['obj_start'] = gene['entities'][1]['start']
        gene['obj_end'] = gene['entities'][1]['end']
        gene['obj_type'] = gene['entities'][1]['type']
        
    else:
        gene['subj_start'] = gene['entities'][1]['start']
        gene['subj_end'] = gene['entities'][1]['end']
        gene['subj_type'] = gene['entities'][1]['type']
        
        gene['obj_start'] = gene['entities'][0]['start']
        gene['obj_end'] = gene['entities'][0]['end']
        gene['obj_type'] = gene['entities'][0]['type']
        

for gene in gene_final:
    if gene['entities'][-1]['end'] > len(gene['token']):
        # print("true")
        gene_final.remove(gene) 

# output_file = r"D:\Lab\project\paper\BioAug-main\datasets-precompute\tarced_f\3_nodis3\inference_origin\mix2_t_f.json"

# with open(output_file, 'w') as f:
#     json.dump(gene_final, f)  
for row in gene_final:
    new_data = {"id": row['origin_id'], "relation": row['relation'], "token": row['token'], "subj_start": row['subj_start'],
                "subj_end": row['subj_end'], "obj_start": row['obj_start'], "obj_end": row['obj_end'],
                "subj_type": row['subj_type'], "obj_type": row['obj_type'], "origin_id": row['old_id'], "old_id2": row['old_id2'],
                "entities": row['entities'], #"mixner_entities": row['mixner_entities'], 
                }
    da_data_list.append(new_data)


# mix_fix_outputname = f"{generation_name}_fix"    
# output_file = r"{}\{}.json".format(args.directory, mix_fix_outputname)
# with open(output_file, 'w') as f:
#     json.dump(da_data_list, f) 
    
combine_data = origin_data + da_data_list  
# In[]

# generation_file_path2 = r"{}\{}.json".format(args.directory, args.input_file2)
with open(os.path.join(args.directory,args.input_file2 + ".json"), 'r') as f:
    generation2 = json.load(f)
    

#檢查實體位置是否有end小於start的情況    
gene_final2 = []
for idx,data in enumerate(generation2):
    # break
    if len(data['entities']) > 0:
        length_flag = True
        for pos in data['entities']:
            if pos['end'] < pos['start']:
                length_flag = False
                break
    if length_flag == False:
        #print("true")
        continue
    
    wrong_data_flag = False
    bidx_list = [t_index for t_index,token in enumerate(data['token']) if token[:2] == "<b"]
    iidx_list = [i_index for i_index,token in enumerate(data['token']) if token[:2] == "<i"]
    b_pair_list = [bidx_list[i:i+2] for i in range(0, len(bidx_list), 2)]
    i_pair_list = [iidx_list[i:i+2] for i in range(0, len(iidx_list), 2)]
    beetween_idx_list = [int((pair[1] + pair[0])/2) for pair in b_pair_list] + [int((pair[1] + pair[0])/2) for pair in i_pair_list]
    ent_type_list = [type['type'] for type in data['entities']]
    for idx,b_pair in enumerate(b_pair_list):
        # break
        # print(idx)
        # 檢查0到第一個是否有出現<i>
        if idx == 0 and bool(set(list(range(0,b_pair[1]))) & set(iidx_list)):
            wrong_data_flag = True
            break
        # 檢查上一個<b>到下一個<b>的區間，檢查最後一個<b>到剩下之token
        elif idx == len(b_pair_list) - 1:
            target_idx_list = list(range(b_pair_list[idx-1][1] + 1, b_pair[0]))
            if bool(set(target_idx_list) & set(iidx_list)):
                target_iidx_list = [value for value in iidx_list if value >= min(target_idx_list) and value <= max(target_idx_list)]
                wrong_data_flag = i_position_2hop_check(target_iidx_list, wrong_data_flag)
            
            target_idx_list = list(range(b_pair[1], len(data['token'])))
            # 如果最後一個<b>之後還有<i>符號
            if bool(set(target_idx_list) & set(iidx_list)):
                target_iidx_list = [value for value in iidx_list if value >= min(target_idx_list) and value <= max(target_idx_list)]
                wrong_data_flag = i_position_2hop_check(target_iidx_list, wrong_data_flag)
        # 檢查上一個<b>到下一個<b>的區間
        else:
             target_idx_list = list(range(b_pair_list[idx-1][1] + 1, b_pair[0]))
             if bool(set(target_idx_list) & set(iidx_list)):
                 target_iidx_list = [value for value in iidx_list if value >= min(target_idx_list) and value <= max(target_idx_list)]
                 wrong_data_flag = i_position_2hop_check(target_iidx_list, wrong_data_flag)
                   
    if wrong_data_flag:
        continue
        
    entities = []
    for idx,b_pair in enumerate(b_pair_list):
        # break
        # if (b_pair[1] + 1) not in iidx_list and (b_pair[1] + 1) not in bidx_list:
        if (b_pair[1] + 1) not in iidx_list:
            ent_data = {
                'type': ent_type_list[idx],
                'start': b_pair[0] + 1,
                'end': b_pair[1],
                }
            entities.append(ent_data)
        elif (b_pair[1] + 1) in iidx_list:
            for i in range(int(b_pair[1] + 1),len(data['token'])):
                if i not in bidx_list and i not in iidx_list and i not in beetween_idx_list:
                    end_pos = i - 1
                    ent_data = {
                        'type': ent_type_list[idx],
                        'start': b_pair[0] + 1,
                        'end': end_pos,
                        }
                    entities.append(ent_data)
                    break
        if idx == 1:
            if data['subj_start'] < data['obj_start']:
                data['subj_start'] = entities[0]['start']
                data['subj_end'] = entities[0]['end']
                data['obj_start'] = entities[1]['start']
                data['obj_end'] = entities[1]['end']
            else:
                data['obj_start'] = entities[0]['start']
                data['obj_end'] = entities[0]['end']
                data['subj_start'] = entities[1]['start']
                data['subj_end'] = entities[1]['end']
        
    data['entities'] = entities
    
    gene_final2.append(data)
    
                
# 修改實體位置(與原生資料集相同包頭包尾)，token資料去除BIO標籤
for gene in gene_final2:
    # break
    gene_m = [element for element in gene['token'] if '<' not in element] 
    for entity in gene['entities']:
        if entity['end'] - entity['start'] == 1:
            start = entity['start']   
            new_start = start - len([tag for tag in gene['token'][:start] if "<" in tag])
            
            entity['start'] = new_start
            entity['end'] = new_start
        else:
            start = entity['start']
            end = entity['end']
            
            new_start = start - len([tag for tag in gene['token'][:start] if "<" in tag])
            new_end = end - len([tag for tag in gene['token'][:end] if "<" in tag]) - 1
            
            entity['start'] = new_start
            entity['end'] = new_end
    gene['token'] = gene_m
    
# 實體位置同步到obj、subj的start,end
for gene in gene_final2:
    # break
    if gene['subj_start'] < gene['obj_start']:
        gene['subj_start'] = gene['entities'][0]['start']
        gene['subj_end'] = gene['entities'][0]['end']
        
        gene['obj_start'] = gene['entities'][1]['start']
        gene['obj_end'] = gene['entities'][1]['end']
        
    else:
        gene['obj_start'] = gene['entities'][0]['start']
        gene['obj_end'] = gene['entities'][0]['end']

        gene['subj_start'] = gene['entities'][1]['start']
        gene['subj_end'] = gene['entities'][1]['end']
        

for gene in gene_final2:
    if gene['entities'][-1]['end'] > len(gene['token']):
        # print("true")
        gene_final2.remove(gene) 


da_data_list2 = []          
for row in gene_final2:
    new_data = {"id": row['origin_id'], "relation": row['relation'], "token": row['token'], "subj_start": row['subj_start'],
                "subj_end": row['subj_end'], "obj_start": row['obj_start'], "obj_end": row['obj_end'],
                "subj_type": row['subj_type'], "obj_type": row['obj_type'], "origin_id": row['old_id']
                }
    da_data_list2.append(new_data)

combine_data = combine_data + da_data_list2
# In[] 
# output_file = r"{}\{}.json".format(args.directory, args.output_file)

with open(os.path.join(args.directory,args.output_file + ".json"), 'w') as f:
    json.dump(combine_data, f) 