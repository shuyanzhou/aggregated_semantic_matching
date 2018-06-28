import h5py
import os
def main(model_name):
	map_name = {"ctx_cnn_w":"query_cnn_w", "ctx_cnn_b":"query_cnn_b",
	"mention_cnn_w":"query_cnn_w", "mention_cnn_b":"query_cnn_b",
	"doc_cnn_w":"entity_cnn_w", "doc_cnn_b":"entity_cnn_b",
	"title_cnn_w":"entity_cnn_w", "title_cnn_b":"entity_cnn_b",
	"combine_w":"combine_w"}
	print(model_name)
	
	new_model = dict_name + os.path.basename(model_name).split(".")[0] + '_new.h5'
	print(new_model)
	fh5 = h5py.File(new_model,"w")
	params = fh5.create_group('params')

	fh5_org = h5py.File(model_name,'r')
	params_org = fh5_org['params']

	for k in params_org.keys():
		#print(k)
		try:
			params[map_name[k]] = params_org[k].value
		except KeyError:
			print("no need to process", new_model)
			fh5.close()
			fh5_org.close()
			os.remove(new_model)
			return
	for k in params.keys():
		print(k)
	
	os.remove(model_name)
	os.rename(new_model, model_name)
	fh5.close()
	fh5_org.close()

#change cxt to ctx because previously spelling error
def change_file_name(dict_name):
	print(len(os.listdir(dict_name)))
	for fname in os.listdir(dict_name):
		if "cxt" in fname:
			os.rename(dict_name + fname, dict_name + fname.replace("cxt","ctx"))
	print(len(os.listdir(dict_name)))

if __name__ == '__main__':
	dict_name = "./data/kbp/result_log/"
	# change_file_name(dict_name)
	fname = os.listdir(dict_name)
	for f in fname :
		#print (f)
		if ".h5" in f and "new" not in f:
			main(dict_name + f)
