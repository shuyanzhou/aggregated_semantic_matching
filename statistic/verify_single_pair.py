import os
#dict_path = "./data/bing_news/rank/"
dict_path = "./data/kbp/rank/"
#fname = "conv_size1ctx-description"
#fname = "conv_size1cxt-description_dif"
aggregate_fname = "kbp1_5_result"
pair_index = 0
def verify_single_file():
	with open(dict_path + fname, "r") as f:
		line = f.readline()
		all_measure = 0
		all_right = 0
		while line:
			tokens = line.strip().split("\t")
			gold = tokens[1]
			predict = " ".join(tokens[2].split(" ")[:-1])
			#print(gold, predict)
			if gold == predict:
				all_right += 1

			all_measure += 1
			line = f.readline()
		print(dict_path + fname)
		print(all_right, all_measure, float(all_right) / all_measure)

def verify_aggregate_file():
	query_dict_single = {}
	single_all_right = 0
	with open(dict_path + fname, "r") as f:
		line = f.readline()
		while line:
			tokens = line.strip().split("\t")
			gold = tokens[1]
			query_id = tokens[0]
			predict = " ".join(tokens[2].split(" ")[:-1])
			if gold == predict:
				single_all_right += 1
			# if query_id in query_dict_single.keys():
			# 	print("WRONG!")
			query_dict_single[query_id] = predict
			line = f.readline()
	
	with open(dict_path + aggregate_fname, "r") as f:
		aggregate_single_all_right = 0
		query_dict_aggregate = {}
		line = f.readline()
		while line:
			tokens = line.strip().split("\t")
			query_id = tokens[0]
			gold = tokens[1]
			predict = tokens[2 * (pair_index + 1)]
			if gold == predict:
				aggregate_single_all_right += 1
			# if query_id in query_dict_aggregate.keys():
			# 	print("WRONG!")
			query_dict_aggregate[query_id] = predict
			line = f.readline()
	for k, v in query_dict_single.items():
		#print(v, query_dict_aggregate[k])
		if v != query_dict_aggregate[k]:
			print(query_id)
	print(single_all_right, aggregate_single_all_right)
	# print(query_dict_single)
	# print(query_dict_aggregate)
if __name__ == '__main__':
	#verify_aggregate_file()
	fname_list = os.listdir(dict_path)
	for fname in fname_list:
		try:
			verify_single_file()
		except:
			pass