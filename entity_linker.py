#from baseWikipediaLinker import PreProcessedQueriesCls
#from cnn_sm import CNNLinker 
from semantic_cnn import SemanticCNN
from semantic_cnn_same_weight import SemanticCNN_Same
from k_nrm import KNRM
from tracking_base_cnn import TrackingBase_CNN
import numpy as np 
import tensorflow as tf 
import time 
import datetime
import sys 
import os 

#import json
import re
import random
import sys
import math 
import h5py
import pickle
from tqdm import tqdm 
import os 
from wikireader import WikiRegexes
import string 
from evaluation import *
import nltk
from config import Config
import copy
from save_rank import *

class EntityConvolutionalLinker(object):
    #num_training_items = 50000000

    def  __init__(self, query_processor, wordembd_vec, config, cnn_weight_setting = "dif"):
        self.wordembd_vec = wordembd_vec
        self.config = config
        self.batch_size = self.config.batch_size
        self.cnn_weight_setting = cnn_weight_setting
        initializer = tf.random_uniform_initializer(-0.01, 0.01)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse = None, initializer=initializer):
                if self.cnn_weight_setting == "dif":
                    self.train_graph = SemanticCNN(self.config, self.wordembd_vec.shape, True, self.config.loss_func)
                elif self.cnn_weight_setting == "same":
                    self.train_graph = SemanticCNN_Same(self.config, self.wordembd_vec.shape, True)
                elif self.cnn_weight_setting == "knrm":
                    self.train_graph = KNRM(self.config, self.wordembd_vec.shape, True)
                elif self.cnn_weight_setting == "tracking":
                    self.train_graph = TrackingBase_CNN(self.config, self.wordembd_vec.shape, True)
        self.dict_path = query_processor.dict_path
        self.train_batches = query_processor.train_batches
        self.train_batch_num = query_processor.train_batch_num
        self.train_dict_org= query_processor.query_dict_train
        self.dev_batches = query_processor.dev_batches
        self.dev_batch_num = query_processor.dev_batch_num
        self.dev_dict = query_processor.query_dict_dev
        self.test_batches = query_processor.test_batches
        self.test_batch_num = query_processor.test_batch_num
        self.test_dict = query_processor.query_dict_test
        
        self.model_name_map = {"Train/Model/CNN/Ctx_CNN_W:0": "ctx_cnn_w", "Train/Model/CNN/Ctx_CNN_b:0": "ctx_cnn_b",
               "Train/Model/CNN/Wdoc_CNN_W:0":"doc_cnn_w", "Train/Model/CNN/Wdoc_CNN_b:0":"doc_cnn_b", 
               "Train/Model/CNN/Mention_CNN_W:0":"mention_cnn_w", "Train/Model/CNN/Mention_CNN_b:0":"mention_cnn_b",
               "Train/Model/CNN/Title_CNN_W:0":"title_cnn_w", "Train/Model/CNN/Title_CNN_b:0":"title_cnn_b",
               "Train/Model/CNN/Query_CNN_W:0": "query_cnn_w", "Train/Model/CNN/Query_CNN_b:0":"query_cnn_b",
               "Train/Model/CNN/Entity_CNN_W:0": "entity_cnn_w", "Train/Model/CNN/Entity_CNN_b:0":"entity_cnn_b",
               "Model/Combine_W:0":"combine_w",
               "Train/Model/Variable_1:0":"mu",
               "Train/Model/Variable_2:0":"sigma",
               "Model/kernel_combine/bias:0":"kernel_bias", "Model/kernel_combine/kernel:0":"kernel_w"} 
        #map name of my model to Feng's
        # self.model_model_name_map = {"Train/Model/CNN/Ctx_CNN_W:0":"Train/Model/Ctx_CNN_W:0",
        # "Train/Model/CNN/Ctx_CNN_b:0": "Train/Model/Ctx_CNN_b:0", 
        # "Train/Model/CNN/Wdoc_CNN_W:0":"Train/Model/Doc_CNN_W:0",
        # "Train/Model/CNN/Wdoc_CNN_b:0":"Train/Model/Doc_CNN_b:0",
        # "Model/Combine_W:0":"Model/Combine_W:0"}
        # self.model_name_map_nf = {"Train/Model/Ctx_CNN_W:0": "ctx_cnn_w", "Train/Model/Ctx_CNN_b:0": "ctx_cnn_b",
        #        "Train/Model/Doc_CNN_W:0":"doc_cnn_w", "Train/Model/Doc_CNN_b:0":"doc_cnn_b", "Model/Combine_W:0":"combine_w"}


    def create_model(self, sess, isTraining=True, model_file=None):
        self.model = self.train_graph
        self.saver = tf.train.Saver()
        if isTraining:
            #add summaries 
            #adding summaries for gradients 
            grad_summaries = []
            for g, v in zip(self.model.train_grad, self.model.train_params):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            loss_summary = tf.summary.scalar("loss", tf.reduce_mean(self.model.loss))
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

            #train summaries
            self.train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            self.train_summary_dir = os.path.join(out_dir, "summaries", "train")
            self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir, sess.graph)
            ##dev summaries
            #self.dev_summary_op = tf.summary.merge([loss_summary])
            #self.dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            #self.dev_summary_writer = tf.summary.FileWriter(self.dev_summary_dir, sess.graph)

            #save parameters
            self.checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            sess.run(tf.global_variables_initializer(), feed_dict = {self.train_graph.embedding_init: self.wordembd_vec})
            #sess.run(set_embedding, feed_dict = {embedding_place: self.wordmatrix})
            print ('Initialized variables and set word embedding!')
        
        # if fsave is not None:
        #     self.saver.restore(sess, fsave)
        #     print("Model restored.")
        #sess.run(tf.global_variables_initializer())
        return self.model
 
    def get_conv_mask(self, mask):
        conv_size = self.config.conv_size
        nquery = mask.shape[0]
        length = mask.shape[1]
        mask_cp = np.zeros((nquery, length - conv_size + 1))
        for i in range(nquery):
            for j in range(length - conv_size + 1):
                span_sum = 0
                for w in range(conv_size):
                    span_sum += mask[i, j + w]
                #all 0, the conv result should be mask
                if span_sum == 0:
                    mask_cp[i, j] = 0
                else:
                    mask_cp[i, j] = 1
        return mask_cp

    def get_pair_mask(self, conv_query_input_mask, conv_entity_input_mask):
        # print(conv_query_input_mask.shape)
        # print(conv_entity_input_mask.shape)
        #assert conv_query_input_mask.shape[0] == conv_entity_input_mask.shape[0]
        pair_mask = np.zeros((conv_query_input_mask.shape[0], conv_query_input_mask.shape[1], conv_entity_input_mask.shape[1]))
        for i in range(pair_mask.shape[0]):
            mask_1 = conv_query_input_mask[i]
            mask_2 = conv_entity_input_mask[i]
            for j in range(pair_mask.shape[1]):
                for k in range(pair_mask.shape[2]):
                    pair_mask[i, j, k] = mask_1[j] * mask_2[k]
        #print(pair_mask)
        return pair_mask
    
    def dev_step(self, sess, cur_batch, dic, log = False):
        if self.config.pair == 'ctx-description':
            query_input = cur_batch.local_ctx
            query_input_mask = self.get_conv_mask(cur_batch.local_ctx_mask)
            entity_input = cur_batch.candidate_entity_description
            entity_input_mask = self.get_conv_mask(cur_batch.description_mask) 
        elif self.config.pair == 'ctx-title':
            query_input = cur_batch.local_ctx
            query_input_mask = self.get_conv_mask(cur_batch.local_ctx_mask)
            entity_input = cur_batch.candidate_entity
            entity_input_mask = self.get_conv_mask(cur_batch.entity_title_mask) 
        elif self.config.pair == 'mention-description':
            query_input = cur_batch.mention
            query_input_mask = self.get_conv_mask(cur_batch.mention_mask)
            entity_input = cur_batch.candidate_entity_description
            entity_input_mask = self.get_conv_mask(cur_batch.description_mask) 
        elif self.config.pair == 'mention-title':
            query_input = cur_batch.mention
            query_input_mask = self.get_conv_mask(cur_batch.mention_mask)
            entity_input = cur_batch.candidate_entity
            entity_input_mask = self.get_conv_mask(cur_batch.entity_title_mask) 
        
        feed_dict = {
            self.train_graph.query_input: query_input,
            self.train_graph.query_mask: query_input_mask,
            self.train_graph.entity_input: entity_input,
            self.train_graph.entity_mask: entity_input_mask,
            #self.train_graph.query_entity_mask: query_entity_mask,
            self.train_graph.y_grouping:cur_batch.grouping_info,
            self.train_graph.y_isgold:cur_batch.gold_answer,
            self.train_graph.dropout_keep_prob:1.0,
        }

        if self.cnn_weight_setting == "knrm":
            query_entity_mask = self.get_pair_mask(query_input_mask, entity_input_mask)
            feed_dict[self.train_graph.query_entity_mask] = query_entity_mask
        
        #start_time = time.time()
        loss, res_vec = sess.run(
         [self.train_graph.loss, self.train_graph.output],
          feed_dict   
         )
        #end_time = time.time()
        #run_time  = end_time - start_time

        #if log:
        # with open(self.dict_path + "result_log/running_log_cpu", "a") as f:
        #     f.write("{}\t{}\n".format(str(query_input.shape[0]), str(run_time)))
        batch_loss = loss 
        batch_link_num = len(cur_batch.grouping_info)
        

        for i in range(len(res_vec)):
            query_id = cur_batch.query_id_offset[i][0]
            candidate_offset = cur_batch.query_id_offset[i][1]
            dic[query_id]['candidate_score'][candidate_offset] = res_vec[i]
        
        return batch_loss, batch_link_num
    
    #for test
    def WritetoFile(self, data, name = "None"):
        with open("./data/conll/use_nf", "a") as f:
            try:
                f.write(str(data.shape))
            except:
                pass
            f.write(name)
            #f.write(str(data.shape))
            f.write(str(data))
            f.write("\n")
            f.write("\n")

    def train_step(self, sess, cur_batch, dic):
        if self.config.pair == 'ctx-description':
            query_input = cur_batch.local_ctx
            query_input_mask = self.get_conv_mask(cur_batch.local_ctx_mask)
            entity_input = cur_batch.candidate_entity_description
            entity_input_mask = self.get_conv_mask(cur_batch.description_mask) 
        elif self.config.pair == 'ctx-title':
            query_input = cur_batch.local_ctx
            query_input_mask = self.get_conv_mask(cur_batch.local_ctx_mask)
            entity_input = cur_batch.candidate_entity
            entity_input_mask = self.get_conv_mask(cur_batch.entity_title_mask) 
        elif self.config.pair == 'mention-description':
            query_input = cur_batch.mention
            query_input_mask = self.get_conv_mask(cur_batch.mention_mask)
            entity_input = cur_batch.candidate_entity_description
            entity_input_mask = self.get_conv_mask(cur_batch.description_mask) 
        elif self.config.pair == 'mention-title':
            query_input = cur_batch.mention
            query_input_mask = self.get_conv_mask(cur_batch.mention_mask)
            entity_input = cur_batch.candidate_entity
            entity_input_mask = self.get_conv_mask(cur_batch.entity_title_mask) 
        
        feed_dict = {
            self.train_graph.query_input: query_input,
            self.train_graph.query_mask: query_input_mask,
            self.train_graph.entity_input: entity_input,
            self.train_graph.entity_mask: entity_input_mask,
            #self.train_graph.query_entity_mask: query_entity_mask,
            self.train_graph.y_grouping:cur_batch.grouping_info,
            self.train_graph.y_isgold:cur_batch.gold_answer,
            self.train_graph.dropout_keep_prob:self.config.dropout_rate
        }

        # fdebug = open("./fdebug.txt", "w+")
        # fdebug.write(str(query_input.shape))
        # fdebug.write(str(entity_input.shape))
        # t1, t2 = sess.run([self.train_graph.query_cnn_reshape_1, self.train_graph.query_cnn_reshape_2], feed_dict)
        # fdebug.write(str(t1.shape))
        # fdebug.write(str(t2.shape))
        if self.cnn_weight_setting == "knrm":
            query_entity_mask = self.get_pair_mask(query_input_mask, entity_input_mask)
            feed_dict[self.train_graph.query_entity_mask] = query_entity_mask
        

        _, step, summaries, loss, res_vec = sess.run(
         [self.train_graph.train, self.train_graph.global_step, 
          self.train_summary_op, self.train_graph.loss, self.train_graph.output],
          feed_dict   
         )

        assert res_vec.shape[0] == query_input.shape[0]
        #self.WritetoFile(res_vec, "matching_score")
        #print("done")
        # t1, t2, t3, t4, t5, t6 = sess.run([self.train_graph.test1, self.train_graph.test2, 
        #        self.train_graph.test3, self.train_graph.test4, self.train_graph.test5, self.train_graph.test6],
        #        feed_dict)
        # self.WritetoFile(t1, "matching_score")
        # self.WritetoFile(t2, "loss_vec")
        # self.WritetoFile(t3, "loss_scalar")
        # self.WritetoFile(t4, "grad")
        # self.WritetoFile(t5, "loss_vec")
        # self.WritetoFile(t6, "loss_scalar")
        # self.WritetoFile(t3, "entity_mention_mask")
        # self.WritetoFile(t3)
        # self.WritetoFile(t4)
        # self.WritetoFile(t5)
        #print(norm_sum)
        batch_loss = loss 
        #batch_loss = 0 
        batch_link_num = len(cur_batch.grouping_info)
        #print(res_vec.shape)
        #print(res_vec)
        for i in range(len(res_vec)):
            query_id = cur_batch.query_id_offset[i][0]
            candidate_offset = cur_batch.query_id_offset[i][1]
            dic[query_id]['candidate_score'][candidate_offset] = res_vec[i]
        return batch_loss, batch_link_num

    
    def do_eval(self, sess, batch_num, batches, dic, save_rank = False, save_top_k = 10, log = False):
        test_loss = 0.0
        test_num = 0.0
        st = 0
        dic_cp = copy.deepcopy(dic)
        
        for i in range(batch_num):
            cur_batch = batches[i]
            cur_loss, cur_num = self.dev_step(sess, cur_batch, dic_cp, log)
            test_loss += cur_loss
            test_num += cur_num
        if save_rank:
            save_rank_file(dic_cp, self.dict_path, "conv_size" + str(self.config.conv_size) + str(self.config.pair) + '_' + str(self.cnn_weight_setting), save_top_k)
        log = []
        tres = ('testing step loss', test_loss/test_num)
        log.append(tres)
        tstate = eval_current_state(dic_cp)
        tstate = ('testing state', tstate)
        print (tstate)
        log.append(tstate)

        f1_res, f1_str = eval_current_state_fahrni(dic_cp)
        teval_res = (f1_str, f1_res['wNIL_KB'], f1_res['wKB_NIL'])
        #print (teval_res)
        log.append(teval_res)
        return f1_res['cKB'] + f1_res['cNIL'], log  

    def saveModel(self, sess, func_op, model_file):
        all_values = sess.run(func_op.train_params)
        all_names = [v.name for v in func_op.train_params]
        model = {name:value for name, value in zip(all_names, all_values)}
        fh5 = h5py.File(model_file+'.h5', 'w')
        #fnames = open(model_file+'_names', 'w')
        params = fh5.create_group('params')
        for k, v in model.items():
            print(k)
        for k, v in model.items():
            #fnames.write(k + '\n')
            params[self.model_name_map[k]] = v
            #params[k]=v
            print (v.shape)
        fh5.close()
        print ('Model saved !')
    
    def restoreModel(self, sess, model_file, only_restore_params=True):
        print(model_file)
        h5_prev = h5py.File(model_file, 'r')
        params = h5_prev['params']
        assign_ops = []
        if only_restore_params:
            #print save info
            for v in params.keys():
                #print (v)
                #print (type(params[v]))
                print ("%s:%s" %(v, str(params[v].value.shape)))
            #print model info
            for v in tf.trainable_variables():
                print("%s:%s" %(v.name, str(v.get_shape())))
            #assign_ops = [v.assign(params[self.model_name_map_nf[self.model_model_name_map[v.name]]].value) for v in tf.trainable_variables() if self.model_name_map_nf[self.model_model_name_map[v.name]] in params.keys() and v.get_shape() == params[self.model_name_map_nf[self.model_model_name_map[v.name]]].value.shape]
            assign_ops = [v.assign(params[self.model_name_map[v.name]].value) for v in tf.trainable_variables() if self.model_name_map[v.name] in params.keys() and v.get_shape() == params[self.model_name_map[v.name]].value.shape]
            print(len(assign_ops))
        else:
            assign_ops = [v.assign(params[v.name]) for v in tf.global_variables() if v.name in params.keys()]
        sess.run(assign_ops)

    def test(self, sess, model_file, log = False):
        #model = self.create_model(sess)
        #model_file = model_file + '.h5'
        #self.restoreModel(sess, model_file)
        self.saver = tf.train.Saver()
        self.saver.restore(sess, "./data/kbp/" + 'model_org/model.ckpt')
        print("Model restore")
        self.do_eval(sess, self.test_batch_num, self.test_batches, self.test_dict, log)

        #save_path = self.saver.save(sess, self.dict_path + '../conll_new/model/model.ckpt')
        #print("Model saved in file: %s" % save_path)


    def train(self, sess, num_iter, batch_num, log_file, model_file):
        train_dict = copy.deepcopy(self.train_dict_org)
        
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))

        f = open(log_file, 'w')
        model = self.create_model(sess)
        # restore_model_file = model_file + '.h5'
        # self.restoreModel(sess, restore_model_file)
        total_loss = 0.0
        total_num = 0.0 
        best_score = 0.0 

        for i in range(num_iter):
            print ('training step %d' %(i))
            ###################
            if i > 0:
                #do some evaluation on dev and test dataset
                print ('\n')
                print ('current training loss %f\n' %(total_loss / total_num))
                f.write('one round loss %f\n' %(total_loss / total_num))
                tstate = eval_current_state(train_dict)
                tstate = ('training state', tstate)
                print (tstate)
                f.write('iteration %d, training %s\n' %(i, str(tstate)))

                #dev and test after every iteration
                flag = False
                score, dev_log = self.do_eval(sess, self.dev_batch_num, self.dev_batches, self.dev_dict)
                if score > best_score:
                    best_score = score
                    self.saveModel(sess, model, model_file)
                    #self.saver.save(sess, self.dict_path + 'model/model.ckpt')
                    flag = True
                    # save_path = self.saver.save(sess, self.checkpoint_prefix + '.ckpt')
                    # print("Model saved in file: %s" % save_path)
                for k in dev_log:
                    f.write('dev %s\n' %(str(k)))
                
                save_rank = flag
                test_score, test_log = self.do_eval(sess, self.test_batch_num, self.test_batches, self.test_dict, save_rank = save_rank)
                for k in test_log:
                    f.write('test %s\n' %(str(k)))

            
            train_dict = copy.deepcopy(self.train_dict_org)
            for j in tqdm(range(self.train_batch_num)):
            #for j in range(1):
                if j != 0 and j % batch_num == 0:
                    print ('\n')
                    print ('current training loss %f\n' %(total_loss / total_num))
                    f.write('one round loss %f\n' %(total_loss / total_num))
                    tstate = eval_current_state(train_dict)
                    tstate = ('training state', tstate)
                    print (tstate)
                    f.write('iteration %d, training %s\n' %(i, str(tstate)))

                    #dev and test after every iteration
                    flag = False
                    score, dev_log = self.do_eval(sess, self.dev_batch_num, self.dev_batches, self.dev_dict)
                    if score > best_score:
                        best_score = score
                        self.saveModel(sess, model, model_file)
                        #self.saver.save(sess, self.dict_path + 'model/model.ckpt')
                        flag = True
                    for k in dev_log:
                        f.write('dev %s\n' %(str(k)))
                    save_rank = flag
                    test_score, test_log = self.do_eval(sess, self.test_batch_num, self.test_batches, self.test_dict, save_rank = save_rank)
                    for k in test_log:
                        f.write('test %s\n' %(str(k)))
                cur_batch = self.train_batches[j]
                if cur_batch is not None:
                    cur_loss, cur_num = self.train_step(sess, cur_batch, train_dict)
                    total_loss += cur_loss
                    total_num += cur_num 
        f.close()
        print("Training Finished!")

        return

