import numpy as np 
import tensorflow as tf 
import time 
import sys 
import os 

#import json
import re
import random
import sys
import math 

from tqdm import tqdm 
import os 
from nn_func import * 
from wikireader import WikiRegexes

class KNRM(object):
	def __init__(self, config, word_embed_dim, isTraining = True):
		self.vocab_size = config.vocab_size
		self.wordembd_dim = config.wordembd_dim 
		self.dim_pos_vec = config.pos_size
		self.hidden_size = config.hidden_size
		self.conv_size = config.conv_size
		self.pair = config.pair

		self.word_embed_dim = word_embed_dim
		self.dropout_rate = config.dropout_rate
		self.grad_norm = config.grad_norm
		#self.enable_train_wordvecs = config.train_word_vec
		if config.pair == 'ctx-description':
			self.query_length = config.local_ctx_len
			self.entity_length = config.wiki_doc_len
		elif config.pair == 'ctx-title':
			self.query_length = config.local_ctx_len
			self.entity_length = config.wiki_title_len
		elif config.pair == 'mention-description':
			self.query_length = config.mention_len
			self.entity_length = config.wiki_doc_len
		elif config.pair == 'mention-title':
			self.query_length = config.mention_len
			self.entity_length =config.wiki_title_len

		#build graph
		self.query_input = tf.placeholder(tf.int32, [None, self.query_length], name = "query_input")
		self.query_mask = tf.placeholder(tf.int32, [None, self.query_length - self.conv_size + 1], name = "query_input_mask")
		self.entity_input = tf.placeholder(tf.int32, [None, self.entity_length], name = "entity_input")
		self.entity_mask = tf.placeholder(tf.int32, [None, self.entity_length - self.conv_size + 1], name = "entity_input_mask")
		self.query_entity_mask = tf.placeholder(tf.int32, [None, self.query_length - self.conv_size + 1, 
												self.entity_length - self.conv_size + 1], name = "query_entity_mask")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_rate")

		#None is of batch_size * candidate_num
		self.y_isgold = tf.placeholder(tf.int32, [None], name = "gold")
		self.y_grouping = tf.placeholder(tf.int32, [None, 3], name = "grouping_start_end_gold")
		
		#embedding
		with tf.device('/cpu:0'):
			self.embedding_init = tf.placeholder(tf.float32, shape = word_embed_dim)
			self.embedding_w = tf.Variable(self.embedding_init, trainable = False)
		with tf.name_scope("embedding"):
			query_embedding = tf.nn.embedding_lookup(self.embedding_w, self.query_input)
			entity_embedding = tf.nn.embedding_lookup(self.embedding_w, self.entity_input)
		with tf.name_scope("CNN"):
			self.query_cnn, _ , _ = CNNMereTensor(query_embedding,
					self.hidden_size,
					self.wordembd_dim,
					self.conv_size,
					self.query_length,
					scope_name = 'Query_CNN')
			self.entity_cnn, _, _ = CNNMereTensor(entity_embedding,
					self.hidden_size,
					self.wordembd_dim,
					self.conv_size,
					self.entity_length,
					scope_name = 'Entity_CNN')

		#kernel pooling layer
		mu_list = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9], dtype = 'float32')
		sigma_list= np.array([1e-3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype = 'float32')
		nkernel = len(mu_list)
		# mu_list = tf.Variable(tf.constant([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]))
		# sigma_list = tf.Variable(tf.constant([1e-3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
		# nkernel = 13
		matching_score= Kernel_Pooling(self.query_cnn, self.entity_cnn, self.query_entity_mask, mu_list, sigma_list, nkernel)
		combine_cosine_output = tf.reshape(matching_score,  [-1])

		def cal_all_cosine(v):
			result = tf.gather(combine_cosine_output, tf.range(v[0], v[1] + 1))
			m = tf.reduce_max(result)
			return m + tf.log(tf.reduce_sum(tf.exp(result - m)))
			#return tf.log(tf.reduce_sum(tf.exp(result)))

		#since y_grouping is of batch_size, we should use map_fn
		#all_cosine_sum = tf.map_fn(expsum, self.y_grouping, dtype = tf.float32)
		all_cosine_sum = tf.map_fn(cal_all_cosine, self.y_grouping, dtype = tf.float32)
		all_cosine_sum = tf.reshape(all_cosine_sum, [-1])
		def cal_gold_cosine(v):
			r = tf.range(v[0], v[1] + 1)
			gold_vector = tf.cast(tf.gather(self.y_isgold, r), tf.float32)
			cs = tf.gather(combine_cosine_output, r)
			gs = tf.multiply(cs, gold_vector) + (1 - gold_vector) * -100000
			m = tf.reduce_max(gs)
			return m + tf.log(tf.reduce_sum(tf.exp(gs - m)))

		gold_cosine = tf.map_fn(cal_gold_cosine, self.y_grouping, dtype = tf.float32)
		gold_cosine = tf.reshape(gold_cosine, [-1])
		
		loss_vec = all_cosine_sum - gold_cosine
		loss_scalar = tf.reduce_sum(loss_vec) / tf.cast(tf.shape(loss_vec)[0], tf.float32)
		
		#self.loss = loss_scalar
		self.loss = tf.reduce_sum(loss_vec)
		self.output = combine_cosine_output

		#test ends here
		if not isTraining:
			return

		#update variables
		tvars = tf.trainable_variables()
		optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, epsilon=1e-06)
		original_grad = tf.gradients(loss_scalar, tvars)

		#this is for test
		#self.grad_norms = [tf.norm(g) for g in original_grad]

		grads, _ = tf.clip_by_global_norm(original_grad, self.grad_norm)
		self.global_step = tf.Variable(0, name = "global_step", trainable = False)
		self.train = optimizer.apply_gradients(
			zip(grads, tvars), 
			global_step = self.global_step)

		self.train_params = tvars
		self.train_grad = original_grad
		self.saver = tf.train.Saver()	
		
		#######for test
		self.test1 = matching_score
		self.test2 = loss_vec
		self.test3 = loss_scalar
		self.test4 = grads
		self.test5 = loss_vec
		self.test6 = loss_scalar
		#self.test4 = self.y_grouping


