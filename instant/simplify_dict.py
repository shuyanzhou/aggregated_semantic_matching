import pickle
dict_path = "./data/conll/"

def main():
	dict_list = ['train_org.pkl', 'dev_org.pkl', 'test_org.pkl']
	for dname in dict_list:
		with open(dict_path + dname, "rb") as f:
			dic = pickle.load(f)
			new_dic = {}
			for k, v in dic.items():
				new_dic[k] = {}
				new_dic[k]['candidate_score'] = v['candidate_score']
				new_dic[k]['candidate_entity'] = v['candidate_entity']
				new_dic[k]['gold_entity'] = v['gold_entity']
		with open(dict_path + dname.split(".")[0] + '.pkl', "wb") as f:
			pickle.dump(new_dic, f)

if __name__ == '__main__':
	main()