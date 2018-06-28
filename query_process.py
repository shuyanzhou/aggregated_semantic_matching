import json
import pickle
import numpy as np
import os
import time
import nltk
from config import Config
import random
class BatchObject(object):
	def __init__(self, query_id_offset, mention, mention_mask,
		local_ctx, local_ctx_mask,
		candidate_entity, entity_title_mask,
		candidate_entity_description, description_mask,
		gold_answer,
		grouping_info):
		self.query_id_offset = query_id_offset
		self.mention = mention
		self.mention_mask = mention_mask
		self.local_ctx = local_ctx
		self.local_ctx_mask = local_ctx_mask
		self.candidate_entity = candidate_entity
		self.entity_title_mask = entity_title_mask
		self.candidate_entity_description = candidate_entity_description
		self.description_mask = description_mask
		self.gold_answer = gold_answer
		self.grouping_info = grouping_info

class QueryProcessor(object):
	def __init__(self, dict_path, query_file, 
		dataset_name, escape_nil_datasset_name, config, isTesting = False):
		self.dict_path = dict_path
		self.query_file = query_file
		#self.split_proportion = split_proportion
		#self.process_state = process_state
		self.config = config
		self.dict_name_list = ["train", "dev", "test"]
		self.isTesting = isTesting
		if dataset_name in escape_nil_datasset_name:
			self.escape_nil = 1
		else:
			self.escape_nil = 0
		
		self.train_batches = []
		self.dev_batches = []
		self.test_batches = []
		
		self.query_dict_all = {}
		self.query_dict_train = {}
		self.query_dict_dev = {}
		self.query_dict_test = {}
		
		self.train_batch_num = 0
		self.dev_batch_num = 0
		self.test_batch_num = 0	
	#random split samples to train/dev/test
	def random_dict(self, start_idx, end_idx, train_state):
		dict_name = self.dict_name_list[train_state]
		dict_file = self.dict_path + dict_name + '.pkl'
		part_dict = {k:v for k, v in self.list_query_dict[start_idx:end_idx]}
		if os.path.exists(dict_file):
			print("{} has existed, no need to save again!".format(dict_file))
		else:
			with open(dict_file, "wb") as f:
				pickle.dump(part_dict, f)
				print("Sucessfully dump {} to file".format(dict_name))		
		return part_dict
	
	#train/dev/test have been well splitted
	def specific_dict(self, train_state):
		dict_name = self.dict_name_list[train_state]
		dict_file = self.dict_path + dict_name + '.pkl'
		part_dict = {}
		part_dict_with_score = {}
		for k, v in self.query_dict_all.items():
			if v['train_state'] == train_state:
				part_dict[k] = v

		#save the dictionary with neccesary information for evaluation
		for k, v in part_dict.items():
			part_dict_with_score[k] = {}
			part_dict_with_score[k]['candidate_score'] = v['candidate_score']
			part_dict_with_score[k]['candidate_entity'] = v['candidate_entity']
			part_dict_with_score[k]['gold_entity'] = v['gold_entity']
		with open(dict_file, "wb") as f:
			pickle.dump(part_dict_with_score, f)
			print("Sucessfully dump {} evaluation dict to file".format(dict_name))

		return part_dict

	def load_split_data(self):
		with open(self.dict_path + "train.pkl", "rb") as f:
			self.query_dict_train = pickle.load(f)
			print("Load train!")
		with open(self.dict_path + "dev.pkl", "rb") as f:
			self.query_dict_dev = pickle.load(f)
			print("Load dev!")
		with open(self.dict_path + "test.pkl", "rb") as f:
			self.query_dict_test = pickle.load(f)	
			print("Load test!")
			return

	def split_data(self, query_file):
		query_file = self.dict_path + query_file
		with open(query_file, "rb") as f:
			self.query_dict_all = pickle.load(f)
			keys = list(self.query_dict_all.keys())
			total = len(keys)
		self.query_dict_train = self.specific_dict(0)
		self.query_dict_dev = self.specific_dict(1)
		self.query_dict_test = self.specific_dict(2)

	def load_batch(self, state):
		if state == 0:
			with open(self.dict_path + "train_batch.pkl", "rb") as f:
				self.train_batches = pickle.load(f)
			print("Sucessfully load train batches!", len(self.train_batches))
			self.train_batch_num = len(self.train_batches)
			return
		elif state == 1:
			with open(self.dict_path + "dev_batch.pkl", "rb") as f:
				self.dev_batches = pickle.load(f)
			print("Sucessfully load dev batches!", len(self.dev_batches))
			self.dev_batch_num = len(self.dev_batches)
			return			
		else:
			with open(self.dict_path + "test_batch.pkl", "rb") as f:
				self.test_batches = pickle.load(f)
			print("Sucessfully load test batches!", len(self.test_batches))
			self.test_batch_num = len(self.test_batches)
			return 


	def get_batch(self, state):
		if state == 0:
			current_dict = self.query_dict_train
			all_batches = self.train_batches
		elif state == 1:
			current_dict = self.query_dict_dev
			all_batches = self.dev_batches
		else:
			current_dict = self.query_dict_test
			all_batches = self.test_batches
		total_query_num = len(current_dict.keys())
		seen_query_num = 0
		flag =  0

		current_batch_size = 0
		total_batch_num = 0
		current_dict_list = list(current_dict.items())
		
		#guarentee the last query is valid
		if state == 0:
			while True:
				random.shuffle(current_dict_list)
				last_query_val = current_dict_list[-1][1]
				situation1 = last_query_val['gold_entity'][0] not in last_query_val['candidate_entity']
				situation2 = self.escape_nil == 1 and state == 0 and last_query_val['gold_entity'][0] == '-NIL-'
				if situation1 or situation2:
					random.shuffle(current_dict_list)
				else:
					break
			
		for k, v in current_dict_list:
			# with open("./data/conll_new_nonil/batch_info", "a", encoding = 'utf-8') as fout:
			# 	fout.write("{}\t{}\n".format(str(k), str(v['candidate_entity'])))
			#print(k, v['candidate_entity'])
			#last batch will not contain batch_size samples, need a flag to show this state
			seen_query_num += 1
			if seen_query_num == total_query_num:
				flag = 1
			#ignore samples whose gold answer is not in candidate_entity list when training
			if state == 0 and v['gold_entity'][0] not in v['candidate_entity']:
				continue
			#ignore nil mention when training
			if self.escape_nil == 1 and state == 0 and v['gold_entity'][0] == '-NIL-':
				continue
			#init
			if current_batch_size == 0:
				mention = np.array([], dtype = 'int32')
				mention_mask = np.array([], dtype = 'int32')
				local_ctx = np.array([], dtype = 'int32')
				local_ctx_mask = np.array([], dtype = 'int32')
				candidate_entity = np.array([], dtype = 'int32')
				candidate_entity_description = np.array([], dtype = 'int32')
				description_mask = np.array([], dtype = 'int32')
				entity_title_mask = np.array([], dtype = 'int32')
				gold_answer = np.array([], dtype = 'int32')
				grouping_info = np.array([], dtype = 'int32')
				#contains a query id and the this candidates' offset in the candidate list
				query_id_offset = np.array([], dtype = 'int32')
				start_idx = 0
			
			try:
				gold_answer_idx = v['candidate_entity'].index(v['gold_entity'][0])
			except ValueError:
				gold_answer_idx = -1
			candid_len = len(v['candidate_entity'])
			for i in range(candid_len):
				mention = np.append(mention, v['mention_wordid'])
				mention_mask = np.append(mention_mask, v['mention_mask'])
				local_ctx = np.append(local_ctx, v['local_ctx_wordid'])
				local_ctx_mask = np.append(local_ctx_mask, v['local_ctx_mask'])
				candidate_entity = np.append(candidate_entity, v['candidate_entity_title_wordid'][i])
				candidate_entity_description = np.append(candidate_entity_description, v['candidate_description_wordid'][i])
				description_mask = np.append(description_mask, v['description_mask'][i])
				entity_title_mask = np.append(entity_title_mask, v['entity_title_mask'][i])
				if i == gold_answer_idx:
					gold_answer = np.append(gold_answer, 1)
				else:
					gold_answer = np.append(gold_answer, 0)
				query_id_offset = np.append(query_id_offset, [k, i])
			
			if gold_answer_idx == -1:
				gold_answer_loc = -1000
			else:
				gold_answer_loc = start_idx + gold_answer_idx
			grouping_info = np.append(grouping_info, [start_idx, start_idx + candid_len - 1, gold_answer_loc])
			start_idx += candid_len
			
			current_batch_size += 1	
			#one batch completed
			if current_batch_size == self.config.batch_size or flag == 1:
				total_batch_num += 1
				#reshape
				mention = mention.reshape(-1, self.config.mention_len)
				mention_mask = mention_mask.reshape(-1, self.config.mention_len)
				local_ctx = local_ctx.reshape(-1, self.config.local_ctx_len)
				local_ctx_mask = local_ctx_mask.reshape(-1, self.config.local_ctx_len)
				candidate_entity = candidate_entity.reshape(-1, self.config.wiki_title_len)
				entity_title_mask = entity_title_mask.reshape(-1, self.config.wiki_title_len)
				candidate_entity_description = candidate_entity_description.reshape(-1, self.config.wiki_doc_len)
				description_mask = description_mask.reshape(-1, self.config.wiki_doc_len)
				
				gold_answer = gold_answer.reshape(-1)
				grouping_info = grouping_info.reshape(-1, 3)
				query_id_offset = query_id_offset.reshape(-1, 2)
				batch_obj = BatchObject(query_id_offset, mention, mention_mask,
					local_ctx, local_ctx_mask,
					candidate_entity, entity_title_mask,
					candidate_entity_description, description_mask, 
					gold_answer, grouping_info)
				# print(batch_obj.query_id_offset)
				# print(batch_obj.mention.shape)
				# print(batch_obj.local_ctx.shape)
				# print(batch_obj.candidate_entity.shape)
				# print(batch_obj.candidate_entity_description.shape)
				# print(batch_obj.gold_answer)
				# print(batch_obj.grouping_info.shape)
				# print(batch_obj.grouping_info)
				all_batches.append(batch_obj)
				current_batch_size = 0
		if state == 0:
			self.train_batch_num = total_batch_num
			with open(self.dict_path + "train_batch.pkl", "wb") as f:
				pickle.dump(all_batches, f)
				print("Sucessfully save train batches!")
		elif state == 1:
			self.dev_batch_num = total_batch_num
			with open(self.dict_path + "dev_batch.pkl", "wb") as f:
				pickle.dump(all_batches, f)
				print("Sucessfully save dev batches!")
		else:
			self.test_batch_num = total_batch_num
			with open(self.dict_path + "test_batch.pkl", "wb") as f:
				pickle.dump(all_batches, f)
				print("Sucessfully save test batches!")

	
	def main(self):
		if not(os.path.exists(self.dict_path + "train_batch.pkl") and os.path.exists(self.dict_path + "dev_batch.pkl") and os.path.exists(self.dict_path + "test_batch.pkl")):
			self.split_data(self.query_file)
			self.get_batch(0)
			self.get_batch(1)
			self.get_batch(2)
		else:
			#now the query_dict_x will only contain necessary information for evaluation
			self.load_split_data()
			self.load_batch(2)
			if not self.isTesting:
				self.load_batch(0)
				self.load_batch(1)

		#for test 
		print(len(self.query_dict_train.keys()))
		print(len(self.query_dict_dev.keys()))
		print(len(self.query_dict_test.keys()))
		print(self.train_batch_num, self.dev_batch_num, self.test_batch_num)

def main():
	config = Config()
	random.seed(12345)
	pickle_processor = PickleProcessor('./data/reddit/', 'query_dict.pkl', 
		[], 1, 'reddit', ['neel_merge_nil', 'ace'], config)

if __name__ == '__main__':
	#np.set_printoptions(threshold = 'nan')
	main()