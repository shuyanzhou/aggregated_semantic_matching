import math
import os
import random
import time
#dict_path = "./data/conll/"
dict_path = "./data/kbp/"
agg_name = "kbp1_5_dif_knrm"
def save_rank_file(queries, dict_path, info, top_k):
	with open(dict_path + "rank/" + info, "w+") as f:
		for k, v in queries.items():
			f.write("{}\t{}\t".format(str(k), v['gold_entity'][0]))
			
			name_score_pair = []
			soft_sum = 0
			#soft max
			# for i in range(len(v['candidate_score'])):
			# 	soft_sum += math.exp(v['candidate_score'][i])
			for i in range(len(v['candidate_score'])):
				name_score_pair.append([v['candidate_entity'][i], v['candidate_score'][i]])
			# for i in range(len(v['candidate_score'])):
			# 	name_score_pair.append([v['candidate_entity'][i], v['candidate_score'][i]])
			sorted_pair = sorted(name_score_pair, key = lambda x: x[1], reverse = True)
			#print(sorted_pair)
			for i in range(min(len(sorted_pair), top_k)):
				f.write("{} {}\t".format(sorted_pair[i][0], sorted_pair[i][1]))
			f.write("\n")

def aggregate_rank(dict_path):
	#escape_file = ["final_file",  "conv_size1", "conv_size2", "conv_size4", "conv_size5"]
	#escape_file = ['final_file', 'mention-title', 'ctx-title', 'mention-description']
	escape_file = ['aggregate_result']
	addition_escape_file = ['_same']
	escape_file += addition_escape_file
	info_dict = {}
	fname_list = os.listdir(dict_path + "rank/")
	f_num = 0
	for i in range(len(fname_list)):
		escape_flag = 0
		for ef in escape_file:
			if ef in fname_list[i]:
				escape_flag = 1
				break
		if escape_flag == 1:
			continue		
		fname = dict_path + "rank/" + fname_list[i]
		print(fname)
		with open(fname, "r") as f:
			f_num += 1
			line = f.readline()
			while line:
				query_id, gold_answer = line.strip().split("\t")[:2]
				name_score_pair = line.strip().split("\t")[2:]
				#rename "NIL" since there are two NIL entity in bing
				NIL_index = -10000
				NIL_flag = 0
				# for p in range(len(name_score_pair)):
				# 	tokens = name_score_pair[p].split(" ")
				# 	name = tokens[:-1]
				# 	name = " ".join(name)
				# 	score = tokens[-1]
				# 	if name == '-NIL-' or name == 'NIL' and NIL_flag == 0:
				# 		NIL_index = p
				# 		NIL_flag = 1

				if query_id not in info_dict.keys():
					info_dict[query_id] = {}
					info_dict[query_id]['gold_answer'] = gold_answer
					info_dict[query_id]['rank'] = []
					info_dict[query_id]['rank'] = [[] for row in range(len(name_score_pair))]
				for p in range(len(name_score_pair)):
					info_dict[query_id]['rank'][p].append(name_score_pair[p])
				line = f.readline()
	print(f_num)
	# print(info_dict['5']['rank'][0])
	# print(info_dict['5']['rank'][1])
	with open(dict_path + "rank/aggregate_result/" + agg_name + str(time.time())[-2:], "w+") as f:
		nil_wrong = 0
		for query_id, info in info_dict.items():
			if len(info_dict[query_id]['rank']) == 1:
				print(query_id, info)
				nil_wrong += 1
				info_dict[query_id]['rank'].append(info_dict[query_id]['rank'][0])
			for i in range(len(info_dict[query_id]['rank'])):
				f.write("{}\t{}\t".format(str(query_id), info['gold_answer']))
				for idx in range(len(info_dict[query_id]['rank'][i])):
					name_score_pair = info_dict[query_id]['rank'][i][idx]
					tokens = name_score_pair.split(" ")
					name = tokens[:-1]
					name = " ".join(name)
					score = tokens[-1]
					if idx == len(info_dict[query_id]['rank'][i]) - 1:
						f.write("{}\t{}\n".format(name, score))
					else:
						f.write("{}\t{}\t".format(name, score))
	print(nil_wrong)
def main():
	random.seed(12345)
	aggregate_rank(dict_path)

if __name__ == '__main__':
	main()