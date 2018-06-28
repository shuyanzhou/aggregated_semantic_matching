class Config(object):
	batch_size = 32
	#context length, default is 30
	local_ctx_len = 40
	mention_len = 10
	doc_len = 100
	
	#wiki article length
	wiki_doc_len = 200
	wiki_title_len = 10
	# wiki_sent_num = 10
	# #this can be modified i GUESS
	# wiki_sent_len = 30
	#query document length
	#convolution window size
	conv_size = None
	#candidate number limit, good for calculation
	max_candi_num = 32
	#hidden unit size
	hidden_size = 300
	#the position information may be useful
	pos_size = 30
	#word vocab
	vocab_size = 200000
	#word_dim = 400
	wordembd_dim = 200
	
	sparse_feature_num = 5
	train_word_vec = False
	dropout_rate = 1.0
	grad_norm = 10.0
	#for kernel
	kernel_num = 15
	pair_list = ["ctx-description", "ctx-title", "mention-description", "mention-title"]
	pair = None

	loss_func = "ML"

