import json
import pickle
nf_file = "../../data/kbp-6w-candi30.json"
sy_file = "./data/kbp/query_dict_words.pkl"

def main():
	with open(nf_file, "r") as f:
		nf_dict = json.load(f)['queries']
	with open(sy_file, "rb") as f:
		sy_dict = pickle.load(f)

	nf_id_candidate = {}
	sy_id_candidate = {}
	for k, v in nf_dict.items():
		for kk ,vv in v.items():
			query_id = k.strip().split("\t")[0]
			if query_id in nf_id_candidate.keys():
				print("NF WARNING!")
			nf_id_candidate[query_id] = [[],[],[]]
			for candid in vv['vals'].keys():
				nf_id_candidate[query_id][0].append(candid)
			nf_id_candidate[query_id][0] = sorted(nf_id_candidate[query_id][0])
			nf_id_candidate[query_id][0]
			nf_id_candidate[query_id][1].append(vv['gold'][0])
			nf_id_candidate[query_id][2].append(int(vv['training']))
	for query_id, v in sy_dict.items():
		if v['doc_id'] in sy_id_candidate.keys():
			print("WARNING!")
		sy_id_candidate[v['doc_id']] = [[], [], []]
		for candid in v['candidate_entity']:
			sy_id_candidate[v['doc_id']][0].append(candid)
		sy_id_candidate[v['doc_id']][0] = sorted(sy_id_candidate[v['doc_id']][0])
		sy_id_candidate[v['doc_id']][0].remove('-NIL-')
		sy_id_candidate[v['doc_id']][1].append(v['gold_entity'][0])
		sy_id_candidate[v['doc_id']][2].append(int(v['train_state']))

	wrong_gold = 0
	wrong_candidate = 0
	right_gold = 0
	right_candidate = 0
	dif_len = 0
	for k, v_nf in nf_id_candidate.items():
		if v_nf[2][0] == 2:
			v_sy = sy_id_candidate[k]
			assert v_sy[2][0] == 2
			if v_nf[0] !=  v_sy[0]:
				if len(v_nf[0])!=len(v_sy[0]):
					dif_len += 1
				total_dif = 0
				print(v_nf[0])
				print(v_sy[0])
				for c in v_sy[0]:
					if c not in v_nf[0]:
						total_dif += 1
				print(len(v_nf[0]), len(v_sy[0]), v_nf[1])
				print(total_dif)
				print("\n")
				wrong_candidate += 1
			else:
				right_candidate += 1
			if v_nf[1] != v_sy[1]:
				wrong_gold += 1
			else:
				right_gold += 1
	print(wrong_gold, right_gold, wrong_candidate, right_candidate, dif_len)

if __name__ == '__main__':
	main()



