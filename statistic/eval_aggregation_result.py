fname = "kbp1_5_dif_knrm94"
with open("./data/kbp/rank/aggregate_result/" + fname + '_result', "r") as f:
	#always_wrong = 0
	all_measure = 0
	all_right = 0
	all_wrong = 0
	#nil_wrong = 0

	#right number of final wrong 
	right_number = [0 for i in range(11)]
	#right number of final right
	wrong_number = [0 for i in range(11)]
	right_id = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
	wrong_id = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
	best_pair_index = 0
	best_pair_win_win = 0
	best_pair_win_loss = 0
	best_pair_loss_win = 0
	best_pair_loss_loss = 0
	best_all_right = 0
	line = f.readline()
	while line:
		tokens = line.strip().split("\t")
		query_id = tokens[0]
		gold = tokens[1]
		# if gold == '-NIL-':
		# 	gold = '-NIL-1'
		predict = tokens[-2]
		predict_answer = tokens[2:-2:2]
		#predict_answer = tokens[2::]
		# if gold not in predict_answer:
		# 	flag1 = 1
		# 	always_wrong += 1
		# 	if ['-NIL-1'] * len(predict_answer) == predict_answer:
		# 		nil_wrong += 1
		# 		print(line)
		# if predict_answer[best_pair_index] == gold:
		# 	best_all_right += 1
		if predict == gold:
			all_right += 1
			num = 0
			for pa in predict_answer:
				if pa == gold:
					num += 1
			#print(num_wrong)
			wrong_number[num] += 1
			wrong_id[num].append(query_id)
			if predict_answer[best_pair_index] == gold:
				best_pair_win_win += 1
			else:
				best_pair_loss_win += 1
		else:
			all_wrong += 1
			num = 0
			for pa in predict_answer:
				if pa == gold:
					num += 1
			# if num == 2:
			# 	print(line)
			right_number[num] += 1
			right_id[num].append(query_id)
			if predict_answer[best_pair_index] == gold:
				best_pair_win_loss += 1
			else:
				best_pair_loss_loss += 1
		all_measure += 1
		line = f.readline()
	print("total_right:", all_right, "total_wrong:", all_wrong, "acc:", all_right / all_measure)
	# print(right_number)
	# print(wrong_number)
	# # print(right_id[4])
	# # print(wrong_id[0])
	# # print(wrong_id[1])
	# print(best_pair_win_win)
	# print(best_pair_win_loss)
	# print(best_pair_loss_win)
	# print(best_pair_loss_loss)
	# print(best_all_right)


	with open("./data/kbp/rank/aggregate_result/" + fname + '_ana', "w+") as f:
		for k, v in right_id.items():
			#print(k, len(v))
			f.write("wrong_right_number")
			f.write(str(k))
			f.write("\t")
			f.write(str(len(v)))
			f.write(str(v))
			f.write("\n")
		for k, v in wrong_id.items():
			#print(k, len(v))
			f.write("right_right_number")
			f.write(str(k))
			f.write("\t")
			f.write(str(len(v)))
			f.write(str(v))
			f.write("\n")