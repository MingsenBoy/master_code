import numpy as np
import shutil
import nltk
import copy
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedTokenizerBase
from datasets import load_dataset
import transformers
transformers.logging.set_verbosity_info()
from typing import Any,Optional,Union
from enum import Enum
from dataclasses import dataclass
import os
# new_working_directory = r'D:\Lab\project\paper\BioAug-main\script'
# os.chdir(new_working_directory)
from utils import get_random_gauss_value, linearize, mask_spacy_entities, add_relations, merge_list
import random
import torch
# import time
import json
from itertools import combinations

def add_relation_triple(sketch, original_text, id, full_train, full_dev):
    if 'train' in id:
        full_data = full_train
    else:
        full_data = full_dev
        

    id_index = id.split("-")[1]
    target_data = full_data[int(id_index)]
    
    subj_entity = ' '.join(target_data['token'][target_data['subj_start']:target_data['subj_end'] + 1])
    obj_entity = ' '.join(target_data['token'][target_data['obj_start']:target_data['obj_end'] + 1])
    relation = target_data['relation']
    
    triple = subj_entity + " " + relation + " " + obj_entity
    
    temp = []
    temp.append(f'</s> {triple}')
    
    sketch += temp
    sketch = merge_list(sketch)
    sketch = sketch.replace(' </s> ', '</s>')
    return sketch
# In[]
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size while training') # Input 16
parser.add_argument('--directory','-dir', default='attn', help='data directory where train and dev files are located')
parser.add_argument('--train_file','-tf', default='attn', help='train file name')
parser.add_argument('--dev_file','-df', default='attn', help='dev file name')
parser.add_argument('--file_name','-f', type=str, default='', help='file name for output')
parser.add_argument('--seed', '-s', type=int, default=-1, help='random seed')
parser.add_argument('--mean', '-mean', type=float, default=0.7, help='mean for gauss prob')
parser.add_argument('--std', '-std', type=float, default=0.1, help='std_dev for gauss prob')
parser.add_argument('--shouldLinearizeAllWords', type=int, default=1, help='linearize mode')
# parser.add_argument('--template_type', type=str, help='dataset template')

args = parser.parse_args()

# args.train_file = "train_processed" 
# args.dev_file = "dev_processed"
# #args.directory = r"D:\Lab\project\paper\paper\scispacy\sciERC_processed\100"
# args.directory = r"D:\Lab\project\paper\BioAug-main\datasets-precompute\tarced_f\3_dis4"
# args.shouldLinearizeAllWords = 1
# args.mean = 0.7
# args.std = 0.1
# args.seed = 42
# args.batch_size = 4
# args.epochs = 10
# args.file_name = "5-100-42--tokenfix"

if not args.seed==-1:
    transformers.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
print(args)

# load the preprocessed dataset with the four kinds of sketches
data_files = {"train": args.train_file+'.json', "validation":args.dev_file+'.json'}
tokenized_dataset = load_dataset(args.directory, data_files=data_files)

full_train = json.load(open(os.path.join(args.directory,'full_train_processed.json'), 'r'))
full_dev = json.load(open(os.path.join(args.directory,'full_dev_processed.json'), 'r'))
#tokenized_dataset = load_dataset(args.directory, data_files=data_files, encoding='cp1252')
print(tokenized_dataset)

# define the inputs and labels for sketch-based reconstruction pre-training
max_input_length = 256
max_target_length = 256

# pretrained checkpoint:
#model_checkpoint = "GanjinZero/biobart-v2-large"
# model_checkpoint = "GanjinZero/biobart-v2-large"
model_checkpoint = 'facebook/bart-large'
#model_checkpoint = 'facebook/bart-base'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
  
new_tokens = ['<b-country>', '<i-country>', '<b-religion>', '<i-religion>', '<b-url>', '<i-url>', '<b-person>', '<i-person>',
              '<b-location>', '<i-location>', '<b-ideology>', '<i-ideology>', '<b-organization>', '<i-organization>',
              '<b-criminal_charge>', '<i-criminal_charge>', '<b-cause_of_death>', '<i-cause_of_death>',
              '<b-misc>', '<i-misc>', '<b-city>', '<i-city>', '<b-state_or_province>', '<i-state_or_province>',
              '<b-date>', '<i-date>', '<b-title>', '<i-title>', '<b-nationality>', '<i-nationality>',
              '<b-number>', '<i-number>', '<b-duration>', '<i-duration>']


# check if the tokens are already in the vocabulary
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(new_tokens))
# In[]
class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )
        
class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"
    
@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    
    tokenizer: PreTrainedTokenizerBase # tokenizer used for encoding the data
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    
    def __call__(self, features, return_tensors=None):
        text = [i['sentence'] for i in features] # sentence
        types = [i['type'] for i in features] # BIO label
        labels = [i['labels'] for i in features] # keyword label
        id = [i['id'] for i in features]
        
        sketch = []
        n_text = []
        # get_random_gauss_value, linearize, mask_spacy_entities, add_relations, merge_list
        for i in range(len(text)): # for ever datapoint in a batch
            # break
            new_text, new_type, new_label = text[i], types[i], labels[i]
            assert len(new_text) == len(new_type) == len(new_label)
            # if len(new_text) != len(new_type) or len(new_text) != len(new_label) or len(new_type) != len(new_label):
            #     continue
            original_text = ' '.join(copy.deepcopy(new_text))
            linearize(new_text, new_label, args.shouldLinearizeAllWords) # linearize實體字 <label> word <label>
            final_y = ' '.join(copy.deepcopy(new_text)) #linearize後的句子
            mask_spacy_entities(new_text, new_type, args.mean, args.std)
            # generated_sketch = add_relations(new_text, original_text, id[i], train_precompute, dev_precompute)# 一句選五筆，
            # generated_sketch = merge_list(new_text)
            generated_sketch = add_relation_triple(new_text, original_text, id[i], full_train, full_dev)
            # print(generated_sketch + "\n")
            sketch.append(generated_sketch)
            n_text.append(final_y)
            
        #model_inputs = tokenizer(sketch, truncation=True, padding='max_length', max_length=1000)
        model_inputs = tokenizer(sketch, max_length=max_input_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(n_text, max_length=max_target_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        
        features = []
        for i in range(len(model_inputs['labels'])):
            features.append({'input_ids': model_inputs['input_ids'][i],
                             'attention_mask': model_inputs['attention_mask'][i],
                             'labels': model_inputs['labels'][i] })
            
        del model_inputs, labels, sketch, n_text, text, types    
        
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
                
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                    
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
            
        return features
    
def compute_metrics(eval_pred):
    return {}
                     
# In[] 
##################################################################
#                     training
##################################################################
batch_size = args.batch_size
num_train_epochs = args.epochs
model_name = model_checkpoint.split("/")[-1]

# load the pretrained weights
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))

# embed_tokens 權重
with torch.no_grad():
    # if 'scierc' in args.directory:
    # if 'tarced_f' in args.directory:
        # new_tokens = ['<b-gene>', '<i-gene>']
    for i, token in enumerate(new_tokens):
        # 使用分詞器獲取詞索引
        token_index = tokenizer.encode(token, add_special_tokens=False)[0]

        # 調整 encoder embeddings
        model.model.encoder.embed_tokens.weight[-(i + 1), :] += model.model.encoder.embed_tokens.weight[token_index, :]

        # 調整 decoder embeddings
        model.model.decoder.embed_tokens.weight[-(i + 1), :] += model.model.decoder.embed_tokens.weight[token_index, :]


# logging_steps = len(tokenized_dataset['train']) // batch_size
if args.directory[-1]!='/':
    args.directory += '/'
#args.train_file = "train_processed"
output_dir = f"{args.directory}{args.train_file}-{args.file_name}"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy = 'epoch',
    save_total_limit = 1,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    fp16 = False, # mixed-precision
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size, #16
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=60,
    remove_unused_columns=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    #compute_metrics=custom_compute_metrics,
)

trainer.train()
save_path = output_dir+"-final"
trainer.save_model(save_path)

shutil.rmtree(output_dir)