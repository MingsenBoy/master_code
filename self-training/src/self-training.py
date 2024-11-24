import os
# new_working_directory = r'D:\Lab\project\paper\GradLRE-master\src'
# os.chdir(new_working_directory)
import sys
import pandas as pd
import transformers
# from utils import get_random_gauss_value, linearize, mask_spacy_entities, add_relations, merge_list
from networks import RelationClassification, LabelGeneration
from transformers import AdamW
import torch.optim as optim
from transformers import BertTokenizer
from transformers import pipeline, AutoTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.distributions import Categorical
import random
import numpy as np
import copy
import time, json
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
from sklearn.metrics import f1_score
from collections import Counter
from mask_words_predict import get_enhance_result
import cbert_finetune
import argparse
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
# In[]
# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='Pytorch For Self-training')
parser.add_argument('--dataset', type=str, default="TACRED",help='dataset name')
parser.add_argument('--directory', help='data directory where pre_data are located')
parser.add_argument('--input_file',help='dataset name')
parser.add_argument('--num_labels', type=int, default=42, help='num labels of the dataset')
parser.add_argument('--max_length', type=int, default=128, help='max token length of the sentence for bert tokenizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--initial_lr', type=float, default=5e-5, help='initial learning rate')
parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')
parser.add_argument('--epochs', type=int, default=6, help='training epochs for labeled data')
parser.add_argument('--total_epochs', type=int, default=10, help='total epochs of the RL learning')
parser.add_argument('--seed_val', type=int, default=42, help='initial random seed value')
parser.add_argument('--unlabel_of_train', type=float, default=0.5, help='unlabeled data percent of the dataset')
args = parser.parse_args()

# args.directory = r"D:\Lab\project\paper\bioaug4re\self-training\data\TACRED"
# args.input_file = r"model2_train5mix2_inference_ver_large"
# # args.label_of_train = 0.1
# args.max_length = 128
# args.unlabel_of_train = 0.5
# args.total_epochs = 10
# args.batch_size  = 16
# args.num_labels = 42
# args.initial_lr = 5e-5
# args.initial_eps = 1e-8
# args.epochs = 6
# args.seed_val = 42

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda")

# label_data_path = f"{args.directory}\\full_train_processed.json"
# all_data_path = f"{args.directory}\\{args.input_file}.json"

with open(os.path.join(args.directory,args.input_file + ".json"), 'r') as f:
    all_data= json.load(f)
    
label_data = []
unlabeled_mix = []

for idx, data in enumerate(all_data):
    # break
    if "old_id2" in data:
        unlabeled_mix.append(copy.deepcopy(data))
    else:
        label_data.append(copy.deepcopy(data))

with open(os.path.join(args.directory,"relation2id.json"),'r') as f:
    relation2id = json.load(f)
    

# In[]
# ------------------------functions----------------------------
# 計算模型預測的準確度
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten() #轉一維
    non_zero_idx = (labels_flat != 0)
    # if len(labels_flat[non_zero_idx])==0:
    #     print("error occur: ", labels_flat)
    #     return 0
    return np.sum(pred_flat[non_zero_idx] == labels_flat[non_zero_idx]) / len(labels_flat[non_zero_idx])


# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# cited: https://github.com/INK-USC/DualRE/blob/master/utils/scorer.py#L26
def score(key, prediction, verbose=True, NO_RELATION=0):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("SET NO_RELATION ID: ", NO_RELATION)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

# ------------------------prepare sentences----------------------------

# Tokenize all of the sentences and map the tokens to thier word IDs.
def pre_processing(sentence_train, sentence_train_label):
    input_ids = []
    attention_masks = []
    labels = []
    e1_pos = []
    e2_pos = []
    # index_list = []

    # Load tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # tokenizer.add_special_tokens({'additional_special_tokens':["<e1>","</e1>","<e2>","</e2>"]})

    # pre-processing sentenses to BERT pattern
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        try:
            # Find e1(id:2487) and e2(id:2475) position
            pos1 = (encoded_dict['input_ids'] == 2487).nonzero()[0][1].item()
            pos2 = (encoded_dict['input_ids'] == 2475).nonzero()[0][1].item()
            e1_pos.append(pos1)
            e2_pos.append(pos2)
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(sentence_train_label[i])
            # index_list.append(i)
        except:
            pass
            #print(sent)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels, device='cuda')
    e1_pos = torch.tensor(e1_pos, device='cuda')
    e2_pos = torch.tensor(e2_pos, device='cuda')
    # index_list = torch.tensor(index_list, device='cuda')
    # w = torch.ones(len(e1_pos), device='cuda')

    # Combine the training inputs into a TensorDataset.
    # train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, w)
    train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos)

    return train_dataset

# dataset = train_dataset
def stratified_sample(dataset, ratio):
    data_dict = {}
    for i in range(len(dataset)):
        if not data_dict.get(dataset[i][2].item()):
            data_dict[dataset[i][2].item()] = []
        data_dict[dataset[i][2].item()].append(i)
    sampled_indices = []
    rest_indices = []
    for indices in data_dict.values():
        random.shuffle(indices)
        sampled_indices += indices[0:int(len(indices) * ratio)]
        rest_indices += indices[int(len(indices) * ratio):int(len(indices) * (ratio + args.unlabel_of_train))]
    # print("****************************************************")
    # print(sampled_indices)
    # print("****************************************************")
    return [Subset(dataset, sampled_indices), Subset(dataset, rest_indices), sampled_indices, rest_indices]

# dataset = train_dataset label_len = label_len
# unlabeled_dataset_now, _, index, _ = stratified_sample(unlabeled_dataset_total, args.unlabel_of_train * 2 / args.total_epochs)
def stratified_sample2(dataset, label_len):
    data_dict = {}
    for i in range(len(dataset)):
        # break
        if not data_dict.get(dataset[i][2].item()):
            data_dict[dataset[i][2].item()] = []
        data_dict[dataset[i][2].item()].append(i)
    sampled_indices = []
    rest_indices = []
    for indices in data_dict.values():
        # break
        # random.shuffle(indices)
        # sampled_indices += indices[0:int(len(indices) * ratio)]
        # rest_indices += indices[int(len(indices) * ratio):int(len(indices) * (ratio + args.unlabel_of_train))]
        sampled_indices += [val for val in indices if val < label_len]
        rest_indices += [val for val in indices if val >= label_len]
    # print("****************************************************")
    # print(sampled_indices)
    # print("****************************************************")
    return [Subset(dataset, sampled_indices), Subset(dataset, rest_indices), sampled_indices, rest_indices]
    # return [Subset(dataset, sampled_indices), sampled_indices]


def add_space(sentence_train, dataset_name):
    if dataset_name == "tacred":
        return sentence_train
    else:
        sentence_train_new = []
        for sentence in sentence_train:
            sentence_train_new.append(sentence.replace("<e1>","<e1> ").replace("</e1>"," </e1>").replace("<e2>","<e2> ").replace("</e2>"," </e2>"))
        return sentence_train_new


# aug_data(sentence_train, sentence_train_label, labeled_indices, masked_model)
# b_index_list = labeled_indices 

def realtion2id(label_name):
    label_id=[]
    with open(os.path.join(args.directory,"relation2id.json"),'r') as f:
        data = json.load(f)
    for i in range(len(label_name)):
        id=data[label_name[i]]
        label_id.append(id)
    return label_id

def Read_TACRED_data(tar_data):
    sentence = []
    label_name = []
    interval=' '
    
    for i in range(len(tar_data)):
        tokens=copy.deepcopy(tar_data[i]["token"])
        subj_start=tar_data[i]["subj_start"]
        subj_end=tar_data[i]["subj_end"]
        obj_start=tar_data[i]["obj_start"]
        obj_end=tar_data[i]["obj_end"]
        relation=tar_data[i]["relation"]
        if subj_start < obj_start:
            tokens.insert(subj_start, '<e1>')
            tokens.insert(subj_end + 2, '</e1>')
            tokens.insert(obj_start + 2, '<e2>')
            tokens.insert(obj_end + 4, '</e2>')
        else:
            tokens.insert(obj_start, '<e1>')
            tokens.insert(obj_end + 2, '</e1>')
            tokens.insert(subj_start + 2, '<e2>')
            tokens.insert(subj_end + 4, '</e2>')
        tokens.insert(0,'[CLS]')
        tokens.insert(-1, '[SEP]')
        tokens=interval.join(tokens)
        sentence.append(tokens)
        label_name.append(relation)
    return sentence,label_name


def cos_dist(x, y):
    return np.dot(x,y)/(np.linalg.norm(x) * np.linalg.norm(y))
# In[]
sentence_train = []
sentence_train_label = []
sentence_train_label_id = []

unlabel_sentence_train = []
unlabel_sentence_train_label = []
unlabel_sentence_train_label_id = []

# Load the dataset.
sentence_train, sentence_train_label = Read_TACRED_data(label_data)
sentence_train_label_id = realtion2id(sentence_train_label)

unlabel_sentence_train, unlabel_sentence_train_label = Read_TACRED_data(unlabeled_mix)
unlabel_sentence_train_label_id = realtion2id(unlabel_sentence_train_label)


label_len = len(sentence_train)
unlabel_len = len(unlabel_sentence_train)


train_dataset = pre_processing(sentence_train, sentence_train_label_id)
train_dataset2 = pre_processing(unlabel_sentence_train, unlabel_sentence_train_label_id)
# define the loss function 
criterion = nn.CrossEntropyLoss()

labeled_dataset, unlabeled_dataset_total, labeled_indices, unlabeled_indices = stratified_sample2(train_dataset, label_len)
unlabeled_dataset_total, unlabeled_dataset_total2, unlabeled_indices, unlabeled_indices2 = stratified_sample2(train_dataset2, unlabel_len)
# labeled_dataset, labeled_indices = stratified_sample(train_dataset, lablel_len)
print(len(labeled_dataset))
print(len(unlabeled_dataset_total))

unlabeled_dataset_list = []
sample_index_list = []
for i in range(args.total_epochs):
    unlabeled_dataset_now, _, index, _ = stratified_sample(unlabeled_dataset_total, args.unlabel_of_train * 2 / args.total_epochs)
    unlabeled_dataset_list.append(unlabeled_dataset_now)
    sample_index_list.append(index)

# Create the DataLoaders for our label and unlabel sets.
labeled_dataloader = DataLoader(
    labeled_dataset,  # The training samples.
    sampler=RandomSampler(labeled_dataset),  # Select batches randomly
    batch_size=args.batch_size  # Trains with this batch size.
)

unlabeled_dataloader_list = []
for i in range(args.total_epochs):
    unlabeled_dataloader_now = DataLoader(
        unlabeled_dataset_list[i],  # The training samples.
        sampler=RandomSampler(unlabeled_dataset_list[i]),  # Select batches randomly
        batch_size=args.batch_size  # Trains with this batch size.
    )
    unlabeled_dataloader_list.append(unlabeled_dataloader_now)

sentence_val = json.load(open(os.path.join(args.directory,"test_sentence.json"), 'r'))
sentence_val_label = json.load(open(os.path.join(args.directory,"test_label_id.json"), 'r'))
val_dataset = pre_processing(sentence_val, sentence_val_label)

validation_dataloader = DataLoader(
    val_dataset,  # The validation samples.
    sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    batch_size=args.batch_size  # Evaluate with this batch size.
)
# Load models
model_teacher = LabelGeneration.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
    # You can increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)
model_teacher = nn.DataParallel(model_teacher)
model_teacher = model_teacher.to(device)

optimizer = AdamW(model_teacher.parameters(),
                   lr=args.initial_lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                   eps=args.initial_eps  # args.adam_epsilon  - default is 1e-8.
                   )
# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
# total_steps = len(labeled_dataloader) * EPOCHS
total_steps1 = len(labeled_dataloader) * (args.epochs)
total_steps2 = 0
for i in range(1, args.total_epochs + 1):
    total_steps2 += len(unlabeled_dataloader_list[i-1])
    total_steps2 += len(labeled_dataloader)
    for j in range(i):
        total_steps2 += len(unlabeled_dataloader_list[j])
total_steps = total_steps1 + total_steps2
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)
# Set the seed value all over the place to make this reproducible.
random.seed(args.seed_val)
np.random.seed(args.seed_val)
torch.manual_seed(args.seed_val)
torch.cuda.manual_seed_all(args.seed_val)

# validation accuracy, and timings.
training_stats = []
# Measure the total training time for the whole run.
total_t0 = time.time()

   # For each epoch...
grad_vector = []
param_x = []
param_y = []
for epoch_i in range(0, args.epochs):
        
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_train_loss = 0
    # Put the model into training mode.
    model_teacher.train()

    # save mixup features
    all_logits = np.array([])
    all_ground_truth = np.array([])

    # For each batch of training data...
    epoch_params = []
    for step, batch in enumerate(labeled_dataloader):
        # break
        batch_params = np.array([])
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            # print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(labeled_dataloader), elapsed))
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)
        # b_index_list = batch[5].to(device)

        model_teacher.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch)
        logits, _ = model_teacher(b_input_ids, 
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels,
                                e1_pos=b_e1_pos,
                                e2_pos=b_e2_pos)
        loss = criterion(logits.view(-1, args.num_labels), batch[2].view(-1))
        total_train_loss += loss.sum().item()
        # Perform a backward pass to calculate the gradients.
        loss.sum().backward()
        for param in model_teacher.parameters():
            if param.grad is not None:
                grad_nmp = param.grad.cpu().numpy().flatten()
                # print(len(grad_nmp))
                if len(grad_nmp) == args.num_labels * 768:
                    param_data = param.data.cpu().numpy().flatten()
                    # print(param_data[-2])
                    # print(param_data[-1])
                    param_x.append(param_data[-2])
                    param_y.append(param_data[-1])
                    batch_params = np.concatenate((batch_params, grad_nmp),axis=None)
        # print("the total param num is: " + str(len(batch_params)))
        epoch_params.append(batch_params)
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model_teacher.parameters(), 1.0)
        optimizer.step()
        
        # Update the learning rate.
        scheduler.step()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        labels_flat = label_ids.flatten()
        all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
        if len(all_logits) == 0:
            all_logits = logits
        else:
            all_logits = np.concatenate((all_logits, logits), axis=0)

    # compute grad cos similarity for each batch
    # for i in range(len(epoch_params)-1):
    #     print(cos_dist(epoch_params[i], epoch_params[i+1]))
    epoch_params = np.array(epoch_params)
    grad_vector.append(np.mean(epoch_params, axis=0))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")
    t0 = time.time()
    # Put the model in evaluation mode
    model_teacher.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    all_prediction = np.array([])
    all_ground_truth = np.array([])
    all_logits = np.array([])
    
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)
        with torch.no_grad():
            (logits, _) = model_teacher(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        e1_pos=b_e1_pos,
                                        e2_pos=b_e2_pos)
        loss = criterion(logits.view(-1, args.num_labels), batch[2].view(-1))
        # Accumulate the validation loss.
        total_eval_loss += loss.sum().item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
        all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
        if len(all_logits) == 0:
            all_logits = logits
        else:
            all_logits = np.concatenate((all_logits, logits), axis=0)
    
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    score(all_ground_truth, all_prediction)

# compute grad cos similarity for each epoch
for i in range(len(grad_vector)-1):
    print(cos_dist(grad_vector[i], grad_vector[i+1]))
grad_vector = np.array(grad_vector)
grad_vector_average = np.mean(grad_vector, axis=0)
# grad_vector_average = grad_vector[-1]

# # Displays final plot
# plt.savefig("param_data_change.png")

train_dataloader = labeled_dataloader
for epoch in range(0, args.total_epochs):


    print("##################################")
    print("teacher model generate pseudo label")
    print("##################################")
    t0 = time.time()
    # Put the model in evaluation mode
    model_teacher.eval()

    # Tracking variables
    total_pseudo_loss = 0
    all_prediction = np.array([])
    all_ground_truth = np.array([])
    all_logits = []

    input_ids = []
    input_mask = []
    gold_labels=[]
    e1_pos = []
    e2_pos = []
    
    # Evaluate data for one epoch
    for batch in unlabeled_dataloader_list[epoch]:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)
        with torch.no_grad():
            (logits, _) = model_teacher(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        e1_pos=b_e1_pos,
                                        e2_pos=b_e2_pos)
        all_logits.append(logits.detach())
        loss = criterion(logits.view(-1, args.num_labels), batch[2].view(-1))
        # Accumulate the validation loss.
        total_pseudo_loss += loss.sum().item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        input_ids.append(b_input_ids)
        input_mask.append(b_input_mask)
        gold_labels.append(b_labels)
        e1_pos.append(b_e1_pos)
        e2_pos.append(b_e2_pos)

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
        all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
    
    # Calculate the average loss over all of the batches.
    avg_pseudo_loss = total_pseudo_loss / len(unlabeled_dataloader_list[epoch])
    # Measure how long the validation run took.
    generate_time = format_time(time.time() - t0)
    score(all_ground_truth, all_prediction)

    input_ids = torch.cat(input_ids, dim=0)
    input_mask = torch.cat(input_mask, dim=0)
    gold_labels = torch.cat(gold_labels, dim=0)
    e1_pos = torch.cat(e1_pos, dim=0)
    e2_pos = torch.cat(e2_pos, dim=0)

    # unlabeled_gold_labels = all_ground_truth
    # unlabeled_first_pseudo_labels = all_prediction

    pseudo_labels = torch.tensor(all_prediction, device='cuda')

    train_add_dataset = train_dataloader.dataset + TensorDataset(input_ids, input_mask, pseudo_labels, e1_pos, e2_pos)
    train_dataloader = DataLoader(
        train_add_dataset,  # The training samples.
        sampler=RandomSampler(train_add_dataset),  # Select batches randomly
        batch_size=args.batch_size  # Trains with this batch size.
    )
    print(len(train_add_dataset))

    # Perform one full pass over the training set.
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.total_epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_train_loss = 0
    # Put the model into training mode.
    model_teacher.train()

    # save mixup features
    all_logits = np.array([])
    all_ground_truth = np.array([])

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            # print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(labeled_dataloader), elapsed))
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)

        model_teacher.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch)
        logits, _ = model_teacher(b_input_ids, 
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels,
                                e1_pos=b_e1_pos,
                                e2_pos=b_e2_pos)
        loss = criterion(logits.view(-1, args.num_labels), batch[2].long().view(-1))
        total_train_loss += loss.sum().item()
        # Perform a backward pass to calculate the gradients.
        loss.sum().backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model_teacher.parameters(), 1.0)
        optimizer.step()
        
        # Update the learning rate.
        scheduler.step()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        labels_flat = label_ids.flatten()
        all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
        if len(all_logits) == 0:
            all_logits = logits
        else:
            all_logits = np.concatenate((all_logits, logits), axis=0)
    
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")
    t0 = time.time()
    # Put the model in evaluation mode
    model_teacher.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    all_prediction = np.array([])
    all_ground_truth = np.array([])
    all_logits = np.array([])
    
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)
        with torch.no_grad():
            (logits, _) = model_teacher(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        e1_pos=b_e1_pos,
                                        e2_pos=b_e2_pos)
        loss = criterion(logits.view(-1, args.num_labels), batch[2].view(-1))
        # Accumulate the validation loss.
        total_eval_loss += loss.sum().item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
        all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
        if len(all_logits) == 0:
            all_logits = logits
        else:
            all_logits = np.concatenate((all_logits, logits), axis=0)
    
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    score(all_ground_truth, all_prediction)
# In[]
# import json
final_list = []
unlabel_train = []
unlabel_train_label = []
unlabel_train_label_id = []
# relation2id = json.load(open('../data/' + args.dataset + '/relation2id.json', 'r'))
rel_dic = list(relation2id.keys())
# unlabeled_mix = json.load(open('../data/' + args.dataset + '/mix2.json', 'r'))
# unlabeled_mix2 = json.load(open('../data/' + args.dataset + '/mix2.json', 'r'))
# unlabel_train = json.load(open('../data/' + args.dataset + '/train_sentencenodis15_mix2.json', 'r'))
# unlabel_train_label = json.load(open('../data/' + args.dataset + '/train_label_iddis_mix2.json', 'r'))
unlabel_train, unlabel_train_label = Read_TACRED_data(unlabeled_mix)
unlabel_train_label_id = realtion2id(unlabel_train_label)
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
for idx in range(len(unlabeled_mix)):
    # break
    encoded_dict = tokenizer.encode_plus(
        unlabel_train[idx],  # Sentence to encode.
        add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
        max_length=args.max_length,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
        truncation=True
    )
    try:
        pos1 = (encoded_dict['input_ids'] == 2487).nonzero()[0][1].item()
        pos2 = (encoded_dict['input_ids'] == 2475).nonzero()[0][1].item()
        
        input_ids = encoded_dict['input_ids'].to(device)
        attention_masks = encoded_dict['attention_mask'].to(device)
        labels = torch.tensor([unlabel_train_label_id[idx]], device='cuda')
        e1_pos = torch.tensor([pos1], device='cuda')
        e2_pos = torch.tensor([pos2], device='cuda')

    except:
        # pass
        # unlabeled_mix.remove()
        # del unlabeled_mix[idx]
        continue

    
    logits, _ = model_teacher(input_ids, 
                            token_type_ids=None,
                            attention_mask=attention_masks,
                            labels=labels,
                            e1_pos=e1_pos,
                            e2_pos=e2_pos)
    
    logits_list = logits[0].tolist()
    rel_label = rel_dic[logits_list.index(max(logits_list))]
    print(rel_label + "****" + unlabeled_mix[idx]["relation"])
    unlabeled_mix[idx]["relation"] = rel_label
    final_list.append(unlabeled_mix[idx])
# In[] 
label_data = label_data + final_list

with open(os.path.join(args.directory,"label_data_self_relabel.json"), 'w') as f:
    json.dump(label_data, f)
# # In[]
# if __name__ == "__main__":
#     sys.exit(main())