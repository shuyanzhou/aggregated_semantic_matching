import json
import pickle
import numpy as np
import os
import time
import random
from tqdm import *
import nltk
from wikireader import WikiRegexes, WikipediaReader
from config import Config
import argparse
import re

with open("/data/data/wiki-400d.bin", "rb") as f:
	header = f.readline()
	# line = f.readline()
	# l = 0
	# while l < 100:
	# 	print(line)
	# 	line = f.readline()
	# 	l += 1
	vocab_size, wordembd_dim = map(int, header.split())
	binary_len = np.dtype('float32').itemsize * wordembd_dim
	word_idx_dict = {}
	wordembd = []
	for line in range(vocab_size):
		word = []
		while True:
			ch = f.read(1)
			if ch == b' ':
				word = b''.join(word)
				print(word)
				break
			if ch != b'\n':
				word.append(ch)
		word_idx_dict[bytes.decode(word)] = line
		wordembd.append(np.fromstring(f.read(binary_len), dtype = 'float32'))
wordembd.append(np.zeros(wordembd_dim, dtype = 'float32'))
word_idx_dict['<UNK>'] = vocab_size
wordembd = np.asarray(wordembd, dtype = 'float32')
vocab_size = vocab_size + 1
wordembd_dim = wordembd_dim
print("Finish load word embeding and sucessfully create word index dictionary!\n")