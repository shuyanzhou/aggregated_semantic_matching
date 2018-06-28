group_1 = [0, 1, 4, 8, 9]
group_2 = [2, 3, 5, 6, 7]
dict_path = "/data/code_zsy/cnn_semantic_new/data/kbp/"
fname = "kbp1_5_dif_knrm94"
def main():
	one_win_two_loss = []
	win_num = []
	total = 0
	with open(dict_path + "rank/aggregate_result/" + fname + "_result", "r") as f:
		line = f.readline()
		while line:
			total += 1
			tokens = line.strip().split("\t")
			gold_answer = tokens[1]
			predict_answer = tokens[2:-2:2]
			#print(predict_answer)
			#w = raw_input()
			assert len(predict_answer) == 10
			group_1_answer = []
			group_2_answer = []
			for i in group_1:
				group_1_answer.append(predict_answer[i])
			for i in group_2:
				group_2_answer.append(predict_answer[i])
			if gold_answer in group_1_answer and gold_answer not in group_2_answer:
				one_win_two_loss.append(int(tokens[0]))
				win_num.append(group_1_answer.count(gold_answer))
			line = f.readline()
	with open(dict_path + "rank/aggregate_result/" + fname + "_cnn_win", "w+") as f:
		for i in range(len(one_win_two_loss)):
			f.write("{}\t{}\n".format(str(one_win_two_loss[i]), str(win_num[i])))
	print(total)
if __name__ == '__main__':
	main()


