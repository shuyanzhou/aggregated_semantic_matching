#!/usr/bin/env python2
# the main file for running
import argparse
import json
import os
import pickle
import numpy as np
import tensorflow as tf 
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
import time
from config import Config
from entity_linker import EntityConvolutionalLinker
from offline_process import OfflineProcessor
from query_process import QueryProcessor
from raw_data_process import *
import random
import sys
#import collections
#from tensorflow.python import debug as tf_debug


def argsp():
    aparser = argparse.ArgumentParser()
    #related files
    aparser.add_argument("--wiki_dump_file", help = 'wiki dump file that contains entity text information', default = '../../data/wiki.xml')
    aparser.add_argument("--wordembd_file", help = 'binary file that contains word embedding vectors', default = '../../data/wiki-400d.bin')
    aparser.add_argument("--escape_nil_dataset_name", help = "datasets that should ignore nil mention when training", default = ['neel', 'ace'])
    #information of process procedure
    aparser.add_argument("--process_state", help = "If the raw input file is tsv file, then 0, else 1", default = 0 )
    aparser.add_argument("--raw_data_format", help = "The raw input file format of this system", default = "json")
    #data info
    aparser.add_argument("--dict_path", help = "dictionary path for required raw input files and generated files", default = './data/kbp/')
    aparser.add_argument("--raw_file", help = 'json file of all queries', default = 'kbp_query.json')
    aparser.add_argument("--dataset_name", help = "the name of the dataset", default = 'kbp')
    #aparser.add_argument("--split_proportion", help = "the proportion of split the dataset into train dev test", default = [])
    #run information
    aparser.add_argument('--isTesting', help = 'whether only restore the model and test', default = False)
    aparser.add_argument('--num_iter', help='number of training iterations', type=int, default=15)
    aparser.add_argument("--conv_size", help = "size of the convolutional window", default = 5)
    aparser.add_argument("--pair_type", help = "the type of information used in CNN model, 0 for ctx-description, 1 for ctx-title, 2 for mention-description, 3 for mention-title",
    						default = 3)
    aparser.add_argument("--batch_size", help = "batch size", default = 32)
    aparser.add_argument("--eval_batch_num", help = "evaluate dev and test for what batch number", default = 300)
    # aparser.add_argument("--method", help = "the method used to measure semantic similarity score, current support: cnn_dif, cnn_same, knrm", 
    #                         default = "cnn_dif")
    #current model : dif same knrm tracking
    aparser.add_argument("--cnn_weight_setting", help = "the weight of cnn", default = 'dif')
    return aparser

def main():
    np.set_printoptions(threshold = np.nan)
    random.seed(12345)
    tf.set_random_seed(12345)
    np.random.seed(12345)
    
    args = argsp().parse_args()
    args.conv_size = int(args.conv_size)
    args.pair_type = int(args.pair_type)
    config = Config()
    config.batch_size = args.batch_size
    config.conv_size = args.conv_size
    config.pair = config.pair_list[args.pair_type]
    print("Current Setting:conv_size:{}, pair:{}".format(str(config.conv_size), config.pair))
    
    if args.process_state == 0:
        raw_data_processor = ProtoProcessor(args.dict_path, args.wiki_dump_file, config)
    else:
        if args.raw_data_format == 'json':
            raw_data_processor = JsonProcessor(args.dict_path, args.wiki_dump_file, args.raw_file, config)
    raw_data_processor.main()
    tsv_file = os.path.basename(raw_data_processor.tsv_file)
    alias_entity_file = os.path.basename(raw_data_processor.alia_entity_file)

    offline_processor = OfflineProcessor(args.dict_path, tsv_file, 
        alias_entity_file, args.wordembd_file, 
        args.process_state, args.raw_data_format, config) 
    offline_processor.main()
    
    config.vocab_size = offline_processor.vocab_size
    config.wordembd_dim = offline_processor.wordembd_dim
    print("The shape of word embedding matrix:{}".format(str(offline_processor.wordembd.shape)))
    
    query_processor = QueryProcessor(args.dict_path, offline_processor.final_query_file, 
        args.dataset_name, args.escape_nil_dataset_name, config, args.isTesting)
    query_processor.main()
    
    print ('Init Model for convolution size {}'.format(str(config.conv_size)))
    queries_exp = EntityConvolutionalLinker(query_processor, offline_processor.wordembd, config, args.cnn_weight_setting)

    sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # gpu_config = tf.ConfigProto()
    # gpu_config.log_device_placement = False
    # gpu_config.gpu_options.allow_growth=True
    # sess = tf.Session(config = gpu_config)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    log_file = args.dict_path + 'result_log/run_log'+ '_conv_size' + str(config.conv_size) + str(config.pair) + '_' + str(args.cnn_weight_setting) + str(time.time())[-3:] + '.txt'
    model_file = args.dict_path + 'result_log/' + 'cnnmodel_conv_size' + str(config.conv_size) + str(config.pair) + '_' + str(args.cnn_weight_setting)
    #model_file = args.dict_path + 'result_log/' + 'uni_base_cnn'
    #if args.num_iter > 0:
    if not args.isTesting:
        print("Current conv_size: ", config.conv_size)
        #model_file = "/data/code_zsy/cnn_semantic_new/data/conll/result_log/cnnmodel_conv_size3cxt-description_dif_for_me"
        queries_exp.train(sess, num_iter=args.num_iter, batch_num = args.eval_batch_num, log_file = log_file, model_file = model_file)
    else:
        #model_file = '/data/code_zsy/input_nf_graph/data/conll/result_log/32_all_same_cnnmodel_conv_size3cxt-description'
        #model_file = './data/kbp/result_log/cnnmodel_conv_size' +  str(config.conv_size) + str(config.pair)
        queries_exp.test(sess, model_file)

if __name__ == '__main__':
    main()
