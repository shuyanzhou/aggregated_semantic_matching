import numpy as np
from tqdm import tqdm
with open("/data/data/wiki-400d.bin", "rb") as f:
	header = f.readline()
	vocab_size, wordembd_dim = map(int, header.split())
	binary_len = np.dtype('float32').itemsize * wordembd_dim
	for line in tqdm(range(vocab_size)):
		word = []
		while True:
			ch = f.read(1)
			if ch == b' ':
				word = b''.join(word)
				break
			if ch != b'\n':
				word.append(ch)
		if bytes.decode(word) != bytes.decode(word).lower():
			print("WRONG!")
# 		self.word_idx_dict[bytes.decode(word)] = line
# 		self.wordembd.append(np.fromstring(f.read(binary_len), dtype = 'float32'))
# self.wordembd.append(np.zeros(wordembd_dim, dtype = 'float32'))
# self.word_idx_dict['<UNK>'] = vocab_size
# self.wordembd = np.asarray(self.wordembd, dtype = 'float32')
# self.vocab_size = vocab_size + 1
# self.wordembd_dim = wordembd_dim
# print("Finish load word embeding and sucessfully create word index dictionary!\n")