import json
import evaluate

import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--input_file', help='data input')
args = parser.parse_args()
print(args)
# In[]
cuda_device = 0
compare_data = json.load(open(args.input_file, 'r'))
# In[]
compare_data_texts = []
for idx, data in enumerate(compare_data):
    if 'stanford_pos' not in data.keys():
        # print(idx)
        compare_data_texts.append(" ".join(data['token']))

perplexity = evaluate.load("perplexity", module_type="metric", device=cuda_device)

input_texts = [s for s in compare_data_texts if s!='']

results = perplexity.compute(model_id='gpt2', predictions=input_texts)
print("perplexity:" + str(round(results["mean_perplexity"], 2)))

