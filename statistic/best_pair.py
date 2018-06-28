import os
dict_path = "./data/conll/result_log/"
fname_list = os.listdir(dict_path)
#conv_size_list = [1,2,3,4,5]
def get_best_pair():
	best_pair = ""
	best_test = ""
	best_test = 0
	for fname in fname_list:
		if not "run_log_conv" in fname:
			continue
		print(fname)
		fname = dict_path + fname
		flag = 0
		best_dev = 0
		with open(fname, "r") as f:
			line = f.readline()
			while line:
				if flag == 1:
					line = f.readline()
					line = f.readline()
					test_score = float(line.strip().split(" ")[-1][:-2])
					#print(test_score)
					if test_score > best_test:
						best_pair = fname
						best_result = line + '\n' + f.readline()
						best_test = test_score
					flag = 0
				else:
					if "dev" in line and "testing state" in line:
						score = float(line.strip().split(" ")[-1][:-2])
						if score > best_dev:
							best_dev = score
							flag = 1
				line = f.readline()
	print(best_pair, best_result)

def compare_same_setting(compare_pair):
	if compare_pair == "conv_size":
		result_fname = open(dict_path + "compare_result/" + "conv_size_result", "a")
		pair_list = [1, 2, 3, 4, 5]
	else:
		result_fname = open(dict_path + "compare_result/" + "info_context_result", "a")
		pair_list = ["ctx-description", "ctx-title", "mention-description", "mention-title"]
	for pair in pair_list:
		for fname in fname_list:
			if compare_pair == "conv_size":
				if not "run_log_conv_size" + str(pair) in fname:
					continue
			else:
				if not (pair in fname and "run_log" in fname):
					continue
			print(fname)
			fname = dict_path + fname
			flag = 0
			best_performance = 0
			acc = 0
			best_test = ""
			with open(fname, "r") as f:
				line = f.readline()
				while line:
					if flag == 1:
						line = f.readline()
						line = f.readline()
						acc = line.strip().split(" ")[-1][:6]
						line = f.readline()
						best_test = line
						flag = 0
					else:
						if "dev" in line and "testing state" in line:
							score = float(line.strip().split(" ")[-1][:-2])
							if score > best_performance:
								best_pair = fname
								best_performance = score
								flag = 1
					line = f.readline()
			tokens = best_test.strip().split(" ")
			#print(tokens)
			prec = tokens[6][:6]
			recall = tokens[11][:6]
			f1 = tokens[14][:6]
			result_fname.write("{}\t{}\t{}\t{}\t{}\n".format(os.path.basename(fname), acc, prec, recall, f1))
		result_fname.write("\n")
	result_fname.close()
def main():
	print(dict_path)
	compare_same_setting("conv_size")
	compare_same_setting("info_context")
	#get_best_pair()

if __name__ == '__main__':
	main()


