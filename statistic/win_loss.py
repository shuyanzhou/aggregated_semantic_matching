import os
def find_only_right(pair_name, compare_pair_list):
	fname = os.listdir(dict_path)
	with open(dict_path + pair_name, "r") as f:
		c1_dict = {}
		line = f.readline()
		while line:
			query_id, gold, predict = line.strip().split("\t")[:3]
			predict = " ".join(predict.split(" ")[:-1])
			if predict == gold:
				c1_dict[query_id] = 1
			else:
				c1_dict[query_id] = 0
			line = f.readline()
	for fn in fname:
		if fn in compare_pair_list:
			with open(dict_path + fn, "r") as f:
				print(fn)
				#c2_dict = {}
				line = f.readline()
				while line:
					query_id, gold, predict = line.strip().split("\t")[:3]
					predict = " ".join(predict.split(" ")[:-1])
					if predict == gold and c1_dict[query_id] == 1:
						c1_dict[query_id] = 0
					line = f.readline()

	for k, v in c1_dict.items():
		if v == 1:
			print(k)




def cal_win_loss(compair_pair):
	fname = os.listdir(dict_path)
	for f1 in fname:
		if compair_pair not in f1:
			continue
		print(f1)
		with open(dict_path + f1, "r") as c1:
			c1_dict = {}
			line = c1.readline()
			while line:
				query_id, gold, predict = line.strip().split("\t")[:3]
				predict = " ".join(predict.split(" ")[:-1])
				if predict == gold:
					c1_dict[query_id] = 1
				else:
					c1_dict[query_id] = 0
				line = c1.readline()
		for f2 in fname:
			if f2 == f1 or compair_pair not in f2:
				continue
			print(f2)
			with open(dict_path + f2, "r") as c2:
				c2_dict = {}
				line = c2.readline()
				while line:
					query_id, gold, predict = line.strip().split("\t")[:3]
					predict = " ".join(predict.split(" ")[:-1])
					if predict == gold:
						c2_dict[query_id] = 1
					else:
						c2_dict[query_id] = 0
					line = c2.readline()
			win_win = 0
			win_loss = 0
			loss_win = 0
			loss_loss = 0
			for k, c1_answer in c1_dict.items():
				c2_answer = c2_dict[k]
				if c1_answer == 0 and c2_answer == 0:
					loss_loss += 1
				elif c1_answer == 0 and c2_answer == 1:
					loss_win += 1
				elif c1_answer == 1 and c2_answer == 1:
					win_win += 1
				elif c1_answer == 1 and c2_answer == 0:
					win_loss += 1
			with open(dict_path + compair_pair + "win_loss", "a") as f:
				#print(dict_path + compair_pair + "win_loss")
				f.write("{}\t{}\n".format(f1, f2))
				f.write("win_win:{}\twin_loss:{}\nloss_win:{}\tloss_loss:{}\n".format(str(win_win), str(win_loss), str(loss_win), str(loss_loss)))
				f.write("\n")
if __name__ == '__main__':
	dict_path = "../data/conll/rank/"
	
	compair_pair = "ctx-description"
	cal_win_loss(compair_pair)

	#pair_name = "conv_size1ctx-description"
	#compare_pair_list = ["conv_size2ctx-description", "conv_size3ctx-description", "conv_size4ctx-description", "conv_size5ctx-description"]

