'''
test whether json file candidate are consistent with my 
'''

import json
import pickle
import copy

json_file = "../../data/kbp-6w-candi30.json"
#pickle_file = "./neel/query_dict_words.pkl"
tsv_file = "./data/kbp/alia_entity.tsv"

def main():
	with open(json_file, "r") as f:
		nf_dict = json.load(f)['queries']
	# with open(pickle_file, "rb") as f:
	# 	sy_dict = pickle.load(f)
	nf_mention_candidate = {}
	sy_mention_candidate = {}
	dif_length_nf = 0
	#check_list = ['MM', 'Air Group Inc.', 'National Union of Mine Workers', ' Source ']
	check_list = ['MM']
	for k, v in nf_dict.items():
		for kk, vv in v.items():
			flag = 0
			if int(vv['training']) == 2:
				local_ctx = kk
				start_idx = local_ctx.index('[')
				end_idx = local_ctx.index(']')
				mention = local_ctx[start_idx + 1:end_idx]
				if mention in check_list:
					print(mention)
					print(kk)
				if mention in nf_mention_candidate.keys():
					candidate_cp = copy.deepcopy(nf_mention_candidate[mention])
					flag = 1
				nf_mention_candidate[mention] = []
				for candid in vv['vals'].keys():
					if candid == 'NIL' or candid == '-NIL-':
						continue
					nf_mention_candidate[mention].append(candid)
				if flag == 1:
					if sorted(candidate_cp) != sorted(nf_mention_candidate[mention]):
						if len(sorted(candidate_cp)) != len(sorted(nf_mention_candidate[mention])):
							dif_length_nf += 1
							# print(candidate_cp)
							# print(nf_mention_candidate[mention])
							# print("\n")
					#assert sorted(candidate_cp) == sorted(nf_mention_candidate[mention])
	
	with open(tsv_file, "r", encoding = 'utf-8') as f:
		line = f.readline()
		while line:
			# flag = 0
			mention, candidate = line.strip().split("\t")
			# if mention in sy_mention_candidate.keys():
			# 	candidate_cp = copy.deepcopy(sy_mention_candidate[mention])
			# 	flag = 1
			# sy_mention_candidate[mention] = []
			
			state_name_pair = candidate.split("|")
			for sn in state_name_pair:
				state = int(sn.split(":")[0])
				name = ":".join(sn.split(":")[1:])
				if name == '-NIL-' or name == 'NIL':
					continue
				if state == 2:
					try:
						sy_mention_candidate[mention].append(name)
					except:
						sy_mention_candidate[mention] = [name, ]
			# if flag == 1:
			# 	assert sorted(candidate_cp) == sorted(sy_mention_candidate[mention])
			line = f.readline()
		# for k, v in sy_mention_candidate.items():
		# 	len_list = len(v)
		# 	sy_mention_candidate[k] = list(set(v))
		# 	len_set = len(sy_mention_candidate[k])
		# 	if len_set != len_list:
		# 		print(len_list, len_set)
		# 	if len_list % len_list != 0:
		# 		print("Question!")

	# for k, v in sy_dict.items():
	# 	if int(v['train_state']) != 2:
	# 		continue 
	# 	flag = 0
	# 	mention = v['mention']
	# 	if mention in sy_mention_candidate.keys():
	# 		candidate_cp = copy.deepcopy(sy_mention_candidate[mention])
	# 		flag = 1
	# 	sy_mention_candidate[mention] = []
	# 	for candid in v['candidate_entity']:
	# 		if candid == '-NIL-' or candid == 'NIL':
	# 			continue
	# 		sy_mention_candidate[mention].append(candid)
	# 	if flag == 1:
	# 		assert sorted(candidate_cp) == sorted(sy_mention_candidate[mention])

	same = 0
	dif_same_length = 0
	dif_length = 0
	no_conform = 0
	not_confirm_list = []
	for mention, nf_candid in nf_mention_candidate.items():
		try:
			sy_candid = sy_mention_candidate[mention]
			if sorted(sy_candid) == sorted(nf_candid):
				same += 1
			else:
				# print(sy_candid)
				# print(nf_candid)
				# print("\n")
				if len(nf_candid) != len(sy_candid):
					# print(sy_candid)
					# print(nf_candid)					
					dif_length += 1
					for c in nf_candid:
						if c not in sy_candid:
							print("WRONG!")
							# print(c)
							# print(sy_candid)
							# print("\n")
				else:
					dif_same_length += 1
					print("\n")
		except KeyError:
			not_confirm_list.append(mention)
			no_conform += 1
	print(same, dif_same_length, dif_length, no_conform)
	print(not_confirm_list)
if __name__ == '__main__':
	main()


