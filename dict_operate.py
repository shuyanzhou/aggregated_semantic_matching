import pickle
import os
import json
class DictOperator(object):
	def save_dict(self, dict_path, file_name, save_dict):
		if os.path.exists(dict_path + file_name):
			print("{} has existed, no need to save it again!".format(file_name))
			return
		file_type = file_name.split(".")[1]
		print("Saving {} ...".format(file_name))
		if file_type == "pkl":
			with open(dict_path + file_name, "wb") as f:
				pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
		elif file_type == "json":
			with open(dict_path + file_name, "w") as f:
				json.dump(save_dict, f)
		#print(len(save_dict.keys()))
		print("Save sucessfully!")