import os
import random

dict_path = "./data/reddit/" 
fname = "reddit_with_ans_standard.tsv"
split_proportion = [0.7, 0.15, 0.15]
def main():
	total_query = 0
	with open(dict_path + fname, "r") as f:
		line = f.readline()
		while line:
			total_query += 1
			line = f.readline()
	train_state = [-1] * total_query
	for i in range(int(total_query * split_proportion[0])):
		train_state[i] = 0
	for i in range(int(total_query * split_proportion[0]), int(total_query * (split_proportion[0] + split_proportion[1]))):
		train_state[i] = 1
	for i in range(int(total_query * (split_proportion[0] + split_proportion[1])), total_query):
		train_state[i] = 2
	random.shuffle(train_state)
	with open(dict_path + fname, "r") as f:	
		with open(dict_path + "query_answer.tsv", "w") as fw:
			current_line = 0
			line = f.readline()
			while line:
				line = line[:-1]
				fw.write("{}\t{}\n".format(line, str(train_state[current_line])))
				current_line += 1
				line = f.readline()
def eval_correct():
	tot = [0, 0, 0]
	with open(dict_path + "query_answer.tsv", "r") as f:
		line = f.readline()
		while line:
			state = int(line.split("\t")[-1])
			tot[state] = tot[state] + 1
			line = f.readline()
	print(tot)
if __name__ == "__main__":
	#main()
	eval_correct()
