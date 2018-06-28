import json
import numpy as np
import os
import re
from wikireader import WikiRegexes, WikipediaReader
from dict_operate import DictOperator
import time
from config import Config
import random
class ProtoProcessor(object):
	def __init__(self, dict_path, wiki_dump_file, config):
		self.dict_path = dict_path
		self.tsv_file = self.dict_path + "query_answer.tsv"
		self.alia_entity_file = self.dict_path + 'alia_entity.tsv'
		self.wiki_dump_file = wiki_dump_file
		self.entity_description_dict = {}
		self.all_entity = set(['-NIL-', 'NIL'])
		self.config = config

	def get_all_entities(self):
		with open(self.alia_entity_file, "r") as f:
			line = f.readline()
			while line:
				_, entity = line.strip().split("\t")
				entity = entity.split("|")
				for a in entity:
					self.all_entity.add(a)
				line = f.readline()
	
	def create_entity_description_dict(self):
		if os.path.exists(self.dict_path + "entity_description_dict.pkl"):
			print("Entity description dictionary has existed!")
			return
		else:
			self.get_all_entities()
			re_pattern = re.compile('[^a-zA-Z0-9_ ]')
			print("Start Saving!")
			page_content = {}
			all_entity_title = [WikiRegexes.convertToTitle(a) for a in self.all_entity]
			class GetWikipediaWords(WikipediaReader, WikiRegexes):
				def readPage(ss, title, content, namespace):
					if namespace != 0:
						return
					tt = ss.convertToTitle(title)
					if tt in all_entity_title:
						ctx = ss._wikiToText(content)
						only_doc = re_pattern.sub('', ctx)
						page_content[tt] = only_doc
					#not recommend, only for convience
					if len(page_content.keys()) == len(self.all_entity):
						print("Find all!")
						DictOperator().save_dict(self.dict_path, "test.pkl", page_content)
			GetWikipediaWords(self.wiki_dump_file).read()

			no_description_entity_num = 0
			for entity in self.all_entity:
				try:
					self.entity_description_dict[entity] = page_content[WikiRegexes.convertToTitle(entity)]
				except KeyError:
					self.entity_description_dict[entity] = '<UNK>'
					no_description_entity_num += 1
			print("{} entities cannot find their description in the wiki_dump file\n".format(no_description_entity_num))
			print("Sucessfully create entity description dictionary!\n")
			DictOperator().save_dict(self.dict_path, "entity_description_dict.pkl", self.entity_description_dict)
	
	def main(self):
		self.create_entity_description_dict()

class JsonProcessor(ProtoProcessor):
	def __init__(self, dict_path, wiki_dump_file, json_file, config):
		super().__init__(dict_path, wiki_dump_file, config)
		self.json_file = self.dict_path + json_file
		self.alias_entity_dict = {}
		self.final_result = []
		self.all_query = {}
		self.is_processed = 0
		if os.path.exists(self.tsv_file) and os.path.exists(self.alia_entity_file):
			self.is_processed = 1
	
	#some local context texts are not formal, too long to contain [mention] within config.local_ctx_len
	def process_local_ctx(self, local_ctx):
		st = 0
		ed = 0
		tokens = local_ctx.split(" ")
		for i in range(len(tokens)):
			if "[" in tokens[i]:
				st = i
			if "]" in tokens[i]:
				ed = i + 1
		if ed > self.config.local_ctx_len:
			tokens = tokens[max(0, st - 20): min(len(tokens), ed + 20)]
			with open("./change_local_ctx", "a") as f:
				f.write(local_ctx)
				f.write("{}\n{}\n".format(local_ctx, " ".join(tokens)))
			return " ".join(tokens)
		else:
			return local_ctx
	
	def process_json(self):
		if self.is_processed == 1:
			print("No need to process!")
			return
		with open(self.json_file, "r") as f:
			all_query = json.load(f)['queries']
		#to see whether a mention's candidate entities have been added, only used when train and dev
		#flag = [{}, {}, {}]
		flag = [{}, {}]
		for k, v in all_query.items():
			doc_id = k.split("\t")[0]
			for kk, vv in v.items():
				if "\t" in kk:
					local_ctx = kk.strip().split("\t")[1:]
					local_ctx = " ".join(local_ctx)
				else:
					local_ctx = kk
				#print(local_ctx)
				local_ctx = self.process_local_ctx(local_ctx)
				if local_ctx.count('[') > 1:
					print(local_ctx)
				start_idx = local_ctx.index('[')
				end_idx = local_ctx.index(']')
				mention = local_ctx[start_idx + 1: end_idx]
				train_state = int(vv['training'])


				if train_state == 2:
					for candid_entity in vv['vals'].keys():
						if candid_entity == 'NIL' or candid_entity == '-NIL-':
							continue
						if mention in self.alias_entity_dict.keys() and [train_state, candid_entity] in self.alias_entity_dict[mention]:
							continue
						else:
							try:
								self.alias_entity_dict[mention].append([train_state, candid_entity])
							except KeyError:
								self.alias_entity_dict[mention] = [[train_state, candid_entity],]	
				else:
					if mention not in flag[train_state].keys():
						flag[int(train_state)][mention] = 1
						for candid_entity in vv['vals'].keys():
							if candid_entity == 'NIL' or candid_entity == '-NIL-':
								continue
							try:
								self.alias_entity_dict[mention].append([train_state, candid_entity])
							except KeyError:
								self.alias_entity_dict[mention] = [[train_state, candid_entity],]
								


				gold_answer = vv['gold'][0]
				self.final_result.append([doc_id, mention, local_ctx, gold_answer, str(train_state)])
				#print(self.final_result[-1])

	def save_processed_files(self):
		if self.is_processed == 1:
			print("No need to save!")
			return
		with open(self.tsv_file, "w", encoding = 'utf-8') as f:
			for result in self.final_result:
				try:
					f.write("{}\t{}\t{}\t{}\t{}\n".format(result[0], result[1], result[2], result[3], result[4]))
				except UnicodeEncodeError:
					print("Error!")
					f.write("{}\t{}\t{}\t{}\t{}\n".format(result[0], result[1], result[2].encode('unicode-escape').decode(), result[3], result[4]))
		with open(self.alia_entity_file, "w", encoding = 'utf-8') as f:
			for k, v in self.alias_entity_dict.items():
				v = list(v)
				#f.write("{}\t".format(k.encode('unicode-escape').decode()))
				f.write("{}\t".format(k))
				for vv in v[:-1]:
					#f.write("{}:{}|".format(str(vv[0]).encode('unicode-escape').decode(), vv[1].encode('unicode-escape').decode()))
					f.write("{}:{}|".format(str(vv[0]), vv[1]))
				#f.write("{}:{}\n".format(str(v[-1][0]).encode('unicode-escape').decode(), v[-1][1].encode('unicode-escape').decode()))
				f.write("{}:{}\n".format(str(v[-1][0]), v[-1][1]))
		print("save tsv file and alia-entity dict successfully!")

	def get_all_entities(self):
		with open(self.alia_entity_file, "r") as f:
			line = f.readline()
			while line:
				_, entity = line.strip().split("\t")
				entity = entity.split("|")
				for a in entity:
					a = ":".join(a.split(":")[1:])	
					self.all_entity.add(a)
				line = f.readline()

	def main(self):
		self.process_json()
		self.save_processed_files()
		self.create_entity_description_dict()

def main():
	random.seed(12345)
	dict_path = "./data/neel/"
	json_file = "neel_query.json"
	wiki_dump_file = '../../data/wiki.xml'
	config = Config()
	json_processor = JsonProcessor(dict_path, wiki_dump_file, json_file, config)
	json_processor.main()

if __name__ == '__main__':
	main()
