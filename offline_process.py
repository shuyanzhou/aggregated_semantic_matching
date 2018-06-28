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
from dict_operate import DictOperator
import copy

class OfflineProcessor(object):
	#alia_entity_file can be a list 
	#dict_path is the dictionary path where these files locate
	def __init__(self, dict_path, tsv_file, alia_entity_file, wordembd_file, process_state, raw_data_format, config):
		self.dict_path = dict_path
		self.tsv_file = self.dict_path + tsv_file
		self.alia_entity_file = self.dict_path + alia_entity_file
		self.wordembd_file = wordembd_file
		#self.wiki_dump_file = wiki_dump_file
		self.process_state = process_state
		self.config = config
		self.raw_data_format = raw_data_format
		self.final_query_file = 'query_dict.pkl'
		if process_state == 1 and self.raw_data_format == 'json' or process_state == 0 and self.raw_data_format == 'use_json':
			self.alia_entity_dict_train = {}
			self.alia_entity_dict_dev = {}
			self.alia_entity_dict_test = {}
			self.alia_entity_dict_list = [self.alia_entity_dict_train, self.alia_entity_dict_dev, self.alia_entity_dict_test]
		else:
			self.alia_entity_dict = {}
		
		self.query_dict = {} #it's the final form
		self.original_query_dict = {} #word form 
		self.mask_info_dict = {}
		self.wordembd = []
		self.word_idx_dict = {}
		self.all_entity = set(['-NIL-'])
		self.entity_title_wordid_dict = {}
		self.entity_description_dict = {}
		self.entity_description_wordid_dict = {}
		self.query_id_local_ctx_dict = {}
		self.re_pattern = re.compile('[^a-zA-Z0-9_ ]')
	
	def process_wordembd_file(self):
		with open(self.wordembd_file, "rb") as f:
			header = f.readline()
			vocab_size, wordembd_dim = map(int, header.split())
			binary_len = np.dtype('float32').itemsize * wordembd_dim
			for line in tqdm(range(vocab_size)):
				word = []
				while True:
					ch = f.read(1)
					if ch == b' ':
						word = b''.join(word)
						#print(word)
						break
					if ch != b'\n':
						word.append(ch)
				self.word_idx_dict[bytes.decode(word)] = line
				self.wordembd.append(np.fromstring(f.read(binary_len), dtype = 'float32'))
		self.wordembd.append(np.zeros(wordembd_dim, dtype = 'float32'))
		self.word_idx_dict['<UNK>'] = vocab_size
		self.wordembd = np.asarray(self.wordembd, dtype = 'float32')
		self.vocab_size = vocab_size + 1
		self.wordembd_dim = wordembd_dim
		print("Finish load word embeding and sucessfully create word index dictionary!\n")

		# with open("word_id_map", "w+", encoding = 'utf-8') as f:
		# 	for k, v in self.word_idx_dict.items():
		# 		f.write("{}\t{}\n".format(k, str(v)))
		# print("Saved!")
	
	def create_alia_entity_dict(self):
		if self.process_state == 1 and self.raw_data_format == 'json' or self.process_state == 0 and self.raw_data_format == 'use_json':
			with open(self.alia_entity_file, "r") as f:
				line = f.readline()
				while line:
					alia, entity = line.strip().split("\t")
					entity = entity.split("|")
					for a in entity:
						if len(a.split(":")) >= 3:
							#print(a)
							state = a.split(":")[0]
							name = ":".join(a.split(":")[1:])
							#print(name)
						else:
							state, name = a.split(":")
						state = int(state)
						self.all_entity.add(name)
						try:
							self.alia_entity_dict_list[state][alia].add(name)
						except KeyError:
							self.alia_entity_dict_list[state][alia] = set([name, ])
					line = f.readline()
			for d in self.alia_entity_dict_list:
				for k, v in d.items():
					d[k] =  list(v)
		else:
			with open(self.alia_entity_file, "r") as f:
				line = f.readline()
				while line:
					alia, entity = line.strip().split("\t")
					entity = entity.split("|")
					for name in entity:
						self.all_entity.add(name)
						try:
							self.alia_entity_dict[alia].add(name)
						except KeyError:
							self.alia_entity_dict[alia] = set([name, ])
					line = f.readline()
			#transfer set to list
			for k, v in self.alia_entity_dict.items():
				self.alia_entity_dict[k] = list(v)	

		print("There are {} entities".format(len(self.all_entity)))
		print("Sucessfully create alia entity dictionary!")

	def get_word_id(self, text, max_len, isSentence = False):
		if isinstance(text, str):
			word_tokens = text.lower().split()
		idx_list = []
		mask = []
		for word in word_tokens[:min(len(word_tokens), max_len)]:
			try:
				idx_list.append(self.word_idx_dict[word.lower()])
				mask.append(1)
			except KeyError:
				idx_list.append(self.word_idx_dict['<UNK>'])
				mask.append(0)
		if len(idx_list) < max_len:
			for i in range(max_len - len(idx_list)):
				idx_list.append(self.word_idx_dict['<UNK>'])
				mask.append(0)
		return idx_list, mask

	def load_entity_description_dict(self):
		with open(self.dict_path + "entity_description_dict.pkl", "rb") as f:
			self.entity_description_dict = pickle.load(f)
			#print(len(self.entity_description_dict.keys()))
			print("Load entity_description_dict!")
			return

	def init_query_info_dict(self, state):
		info_dict = {}
		if state == 0:
			info_dict['candidate_entity'] = []
			info_dict['candidate_entity_title_wordid'] = []
			info_dict['local_ctx_wordid'] = []
			info_dict['candidate_score'] = []
			info_dict['doc_id'] = ''
			info_dict['gold_entity'] = []
			info_dict['mention_wordid'] = []
			info_dict['candidate_description_wordid'] = []
			info_dict['train_state'] = 0
			info_dict['mention_mask'] = []
			info_dict['local_ctx_mask'] = []
			info_dict['description_mask'] = []
			info_dict['entity_title_mask'] = []
		elif state == 1:
			info_dict['candidate_entity'] = []
			info_dict['candidate_entity_title'] = []
			info_dict['local_ctx'] = []
			info_dict['doc_id'] = ''
			info_dict['gold_entity'] = []
			info_dict['mention'] = []
			info_dict['candidate_description'] = []
			info_dict['train_state'] = 0		
		return info_dict

	def process_tsv_file(self):
		with open(self.tsv_file, "r", encoding = "utf-8") as f:
			line = f.readline()
			query_num = 0
			no_dscpt = 0
			total_entity = 0
			while line:
				doc_id, mention, local_ctx, gold_entity, train_state = line.strip().split("\t")		

				self.query_dict[query_num] = self.init_query_info_dict(0)
				self.original_query_dict[query_num] = self.init_query_info_dict(1)

				self.query_dict[query_num]['train_state'] = int(train_state)
				self.original_query_dict[query_num]['train_state'] = int(train_state)

				self.query_dict[query_num]['doc_id'] = str(doc_id)	
				self.original_query_dict[query_num]['doc_id'] = str(doc_id)	

				mention_wordid, mention_mask = self.get_word_id(mention, self.config.mention_len, False)
				self.query_dict[query_num]['mention_wordid'].append(mention_wordid)
				self.query_dict[query_num]['mention_mask'].append(mention_mask)
				self.original_query_dict[query_num]['mention'] = mention

				only_local_ctx = self.re_pattern.sub(' ', local_ctx)
				local_ctx_wordid, local_ctx_mask = self.get_word_id(only_local_ctx, self.config.local_ctx_len, True)
				self.query_dict[query_num]['local_ctx_wordid'].append(local_ctx_wordid)
				self.query_dict[query_num]['local_ctx_mask'].append(local_ctx_mask)
				self.original_query_dict[query_num]['local_ctx'] = local_ctx

				self.query_dict[query_num]['gold_entity'].append(gold_entity)
				self.original_query_dict[query_num]['gold_entity'].append(gold_entity)

				if self.process_state == 1 and self.raw_data_format == 'json' or self.process_state == 0 and self.raw_data_format == 'use_json':
					try:
						candidate_entity_list = self.alia_entity_dict_list[int(train_state)][mention]
					except KeyError:
						candidate_entity_list = []
				else:
					try:
						candidate_entity_list = self.alia_entity_dict[mention]
					except KeyError:
						candidate_entity_list = []
				addition_entity = ['NIL']
				# if str(train_state) == '0':
				# 	if len(candidate_entity_list) < 10 and len(candidate_entity_list) > 0:
				# 		addition_entity += random.sample(self.all_entity, 10 - len(candidate_entity_list))
				#for entity in candidate_entity_list + ['-NIL-']:
				for entity in candidate_entity_list + addition_entity:
					self.query_dict[query_num]['candidate_entity'].append(entity)
					self.original_query_dict[query_num]['candidate_entity'].append(entity)


					wiki_entity_title = WikiRegexes.convertToTitle(entity)
					#wiki_entity_title = entity
					wiki_entity_title_wordid, wiki_entity_title_mask = self.get_word_id(wiki_entity_title, self.config.wiki_title_len, False)
					self.query_dict[query_num]['candidate_entity_title_wordid'].append(wiki_entity_title_wordid)
					self.query_dict[query_num]['entity_title_mask'].append(wiki_entity_title_mask)
					self.original_query_dict[query_num]['candidate_entity_title'].append([entity, wiki_entity_title])	

					total_entity += 1
					if entity not in self.entity_description_dict.keys():
						no_dscpt += 1
						description = "<UNK>"
					else:
						description = self.entity_description_dict[entity]
					#shorten the description to save process time
					description = description[:min(len(description), self.config.wiki_doc_len * 15)]
					entity_description_wordid, description_mask = self.get_word_id(description, self.config.wiki_doc_len, True)
					self.query_dict[query_num]['candidate_description_wordid'].append(entity_description_wordid)
					self.query_dict[query_num]['description_mask'].append(description_mask)
					self.original_query_dict[query_num]['candidate_description'].append(description)
					#self.query_dict[query_num]['description_mask'].append(self.get_conv_mask(description_mask))

					self.query_dict[query_num]['candidate_score'].append(0)
				line = f.readline()
				query_num += 1

		DictOperator().save_dict(self.dict_path, self.final_query_file, self.query_dict)
		DictOperator().save_dict(self.dict_path, self.final_query_file.split(".")[0] + "_words.pkl", self.original_query_dict)
		print(no_dscpt, total_entity)

	def main(self):
		self.process_wordembd_file()
		if os.path.exists(self.dict_path + self.final_query_file) and os.path.exists(self.dict_path + self.final_query_file.split(".")[0] + "_words.pkl"):
			print("No need to further process!")
		else:	
			self.create_alia_entity_dict()
			self.load_entity_description_dict()
			self.process_tsv_file()


def argsp():
	aparser = argparse.ArgumentParser()
	aparser.add_argument("--process_state", help = "whether train dev test have been splitted into different tsv file, 0 for normal, 1 for special data", 
                            default = 1)
	aparser.add_argument("--dict_path", help = 'dictionary path for related files', default = './data/test/')
	aparser.add_argument("--tsv_file", help = 'raw tsv file', default = 'query_answer.tsv')
	aparser.add_argument("--alia_entity_file", help = 'alias and their candidate entities', default = 'alia_entity.tsv')
	#aparser.add_argument("--wiki_dump_file", help = 'wiki dump file', default = '../../data/wiki.xml')
	aparser.add_argument("--wordembd_file", help = 'binary file that contains word embedding vectors', default = '../../data/wiki-400d.bin')
	return aparser

def main():
	config = Config()
	args = argsp().parse_args()
	preprocessor = OfflineProcessor(args.dict_path, args.tsv_file, 
		args.alia_entity_file, args.wordembd_file, 
		args.process_state, config)
	preprocessor.main()

if __name__ == '__main__':
	main()
