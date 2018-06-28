import re
from collections import defaultdict

re_pattern = re.compile('[^a-zA-Z0-9_\[\] ]')
def get_word_idx_map(fname):
    with open(fname, "r", encoding = "utf-8") as f:
        word_idx_map = {}
        unk_idx = -1
        for line in f:
            word, idx = line.strip().split("\t")
            word_idx_map[word] = idx
            if word == "<UNK>":
                unk_idx = idx
                print("[INFO] unk idx:{}".format(unk_idx))
    print("[INFO] number of word:{}".format(len(word_idx_map.keys())))

    return word_idx_map, unk_idx

#get all entity id that has embedding
def get_exist_entity_embedding(fname):
    with open(fname, "r", encoding = "utf-8") as f:
        exist_entity_wikiid = set()
        for line in f:
            entity_wikiid, _ = line.strip().split("\t")
            exist_entity_wikiid.add(entity_wikiid)
    print("[INFO] number of entities that have embedding:{}".format(len(exist_entity_wikiid)))
    return exist_entity_wikiid

def get_entity_wikiid_map(fname):
    with open(fname, "r", encoding = "utf-8") as f:
        entity_wikiid_map = {}
        for line in f:
            idx, _, name = line.strip().split("\t")
            entity_wikiid_map[name] = idx
    print("[INFO] number of entities in original mapping:{}".format(len(entity_wikiid_map.keys())))
    return entity_wikiid_map

def get_redirect(fname):
    with open(fname, "r",encoding = "utf-8") as f:
        redirect_map = {}
        for line in f:
            tokens = line.strip().split("\t")
            if len(tokens) != 2:
                print("[ERROR]" +  line)
                continue
            org, red = tokens
            redirect_map[org] = red
    return redirect_map

def get_candidate_map(fname):
    map_list = [defaultdict(set), defaultdict(set), defaultdict(set)] #mention and its candidate WIKIID
    id_not_exist = set()
    not_exist_in_yamada = set()
    all_entity = set()
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            mention, candidates = line.strip().split("\t")
            for candid in candidates.split("|"):
                tokens = candid.split(":")
                candid_name = ":".join(tokens[1:])
                candid_name = redirect_map.get(candid_name, candid_name)
                all_entity.add(candid_name)
                if candid_name not in entity_wikiid_map.keys():
                    id_not_exist.add(candid_name)
                else:
                    entity_wikiid = entity_wikiid_map[candid_name]
                    map_list[int(tokens[0])][mention].add(entity_wikiid)
                    if entity_wikiid not in exist_entity_wikiid:
                        not_exist_in_yamada.add(candid_name)
    print("[INFO] total number of entities:{}".format(len(all_entity)))
    print("[INFO] number of entities that don't have wikiid:{}".format(len(id_not_exist)))
    print("[INFO] number of entities that don't have embeddings:{}".format(len(not_exist_in_yamada)))
    print(map_list[0]["Communist Party of Spain"])
    print(map_list[1]["Communist Party of Spain"])
    print(map_list[2]["NCTA"])

    return map_list

def text2idx(text):
    st = 0 
    ed = 0
    tokens = re_pattern.sub("", text).split()
    for i in range(len(tokens)):
        if "[" in tokens[i]:
            st = i
        if "]" in tokens[i]:
            ed = i
    tokens[ed] = tokens[ed][:-1]
    tokens[st] = tokens[st][1:]
    tokens = tokens[max(0, st - 20):min(len(tokens), ed + 20)]
    # print(tokens)
    idx_list = []
    for t in tokens[:min(len(tokens), 40)]:
        idx_list.append(word_idx_map.get(t.lower(), unk_idx))
    if len(tokens) < 40:
        for i in range(40-len(tokens)):
            idx_list.append(unk_idx)
    assert(len(idx_list) == 40)

    return idx_list

def write_results(fname, fsave):
    no_candid_num = 0
    id_not_exist =set()
    not_exist_in_yamada = set()
    all_ans = set()
    dset_num = []
    with open(fname, "r", encoding = "utf-8") as fin:
        with open(fsave, "w+", encoding = "utf-8") as fout:
            for line in fin:
                _, mention, ctx, ans, dset = line.strip().split("\t")
                dset = int(dset)
                ctx_idx_list = text2idx(ctx)
                candid_list = map_list[dset].get(mention, [])
                if len(candid_list) == 0:
                    no_candid_num += 1
                    candid_list = set(["EMPTYCAND"])
                ans = redirect_map.get(ans, ans)
                all_ans.add(ans)
                if ans not in entity_wikiid_map.keys():
                    id_not_exist.add(ans)
                    ans_wikiid = '-1'
                else:
                    ans_wikiid = entity_wikiid_map[ans]
                    if ans_wikiid not in exist_entity_wikiid:
                        not_exist_in_yamada.add(ans)
                # print(ctx_idx_list,"||", candid_list, "||", ans_wikiid)
                s = " ".join(ctx_idx_list) + "\t" + " ".join(list(candid_list)) + "\t" + ans_wikiid + "\t" + str(dset) + "\n"
                fout.write(s)

    print("[INFO] total number of entities:{}".format(len(all_ans)))
    print("[INFO] number of entities that don't have wikiid:{}".format(len(id_not_exist)))
    print("[INFO] number of entities that don't have embeddings:{}".format(len(not_exist_in_yamada)))
    print(not_exist_in_yamada)

if __name__ == "__main__":
    word_idx_map, unk_idx = get_word_idx_map("../yamada/word_id_map_300.tsv")
    entity_wikiid_map = get_entity_wikiid_map("../yamada/en_id.tsv")
    redirect_map = get_redirect("../yamada/en_redirect.tsv")
    exist_entity_wikiid = get_exist_entity_embedding("../yamada/entity_embedding_300.tsv")
    map_list = get_candidate_map("./data/kbp/alia_entity.tsv")
    write_results("./data/kbp/query_answer.tsv", "./data/fortg/new_kbp_data_300.tsv")

