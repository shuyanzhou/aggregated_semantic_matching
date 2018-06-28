import pickle
from query_process import *
import sys

def tranfer_to_2(fname, fnew):
	with open(fname, "rb") as f:
		dic = pickle.load(f)

	with open(fnew, "wb") as f:
		pickle.dump(dic, f, protocol = 2)
	print(fname)

if __name__ == "__main__":
	fname = ["./data/kbp/train_batch.pkl", "./data/kbp/dev_batch.pkl", "./data/kbp/test_batch.pkl"]
	fnew = ["/home/v-shuyz/train_batch_2.pkl", "/home/v-shuyz/dev_batch_2.pkl", "/home/v-shuyz/test_batch_2.pkl"]
	for i in range(3):
		tranfer_to_2(fname[i], fnew[i])