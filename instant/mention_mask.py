tsv_file = "./data/kbp/query_answer.tsv"
import re
import copy
import json

def get_mention_mask(text, local_ctx_len, conv_len):
    mention_mask = [0] * local_ctx_len
    st = 0
    ed = 0
    get_words_with_mention = re.compile('[^a-zA-Z0-9\[\] ]')
    tokens = get_words_with_mention.sub(' ', text).split()
    print(tokens)
    for i in range(len(tokens)):
        if "[" in tokens[i]:
            st = i
        if "]" in tokens[i]:
            ed = i + 1
    for i in range(st, ed):
        mention_mask[i] =  1
    conv_mention_mask = copy.deepcopy(mention_mask)
    padding = int(conv_len / 2)
    for i in range(padding):
        conv_mention_mask.insert(0, 0)
        conv_mention_mask.append(0)
    conv_mention_mask_ = copy.deepcopy(conv_mention_mask)
    for i in range(padding, len(conv_mention_mask) - padding):
        if conv_mention_mask[i] == 1:
            for j in range(max(0, i - padding), min(local_ctx_len, i + padding) + 1):
                conv_mention_mask_[j] = 1
    conv_mention_mask = conv_mention_mask_[padding * 2:len(conv_mention_mask) - padding * 2]
    print(conv_mention_mask)
    return conv_mention_mask

# with open(tsv_file, "r") as f:
# 	line = f.readline()
# 	while line:
# 		_, _, ctx, _, _ = line.strip().split("\t")
# 		cmm = get_mention_mask(ctx, 40, 3)
# 		line = f.readline()

alia_entity_file = "./data/kbp/alia_entity_train.tsv"

with open(alia_entity_file, "r") as f:
    line = f.readline()
    while line:
        s = set()
        mention, entity = line.strip().split("\t")
        entity = entity.strip().split("|")
        if len(entity) > 100:
            print(len(entity))
        line = f.readline()

# fname = "./data/kbp/kbp_6w_candi30.json"
# with open(fname, "r") as f:
#     num = 0
#     d = json.load(f)['queries']
#     for k, v in d.items():
#         for kk, vv in v.items():
#             l = len(vv['vals'].keys())
#             if l > 60:
#                 print(l)
#                 num += 1
# print(num)


