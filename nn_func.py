import tensorflow as tf
import numpy as np 
import sys 
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from config import Config

def CNNTensor(
    input, hidden_size, #hidden size means the number of filters
    word_dim, conv_size, length,
    scope_name='CNN', reshape = True):
    filter_shape = [conv_size, word_dim, 1, hidden_size]
    con_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name=scope_name+'_W')
    con_b = tf.Variable(tf.constant(0.0, shape=[hidden_size]), name=scope_name+'_b')
    #input = batch * doc_len * word_embd - > () * 1
    #context_cnn = batch * (doc_len - window_size + 1) * 1 * hidden_size
    context_cnn = tf.nn.conv2d(
        tf.expand_dims(input, -1),
        con_W,
        strides=[1, 1, 1, 1],
        padding='VALID',
        name=scope_name+'_cnn'    
    )
    context_cnn_bias = tf.nn.relu(tf.nn.bias_add(context_cnn, con_b), name=scope_name + 'relu')
    # context_bias = tf.nn.bias_add(context_cnn, con_b)
    # context_cnn_bias = tf.maximum(context_bias, -.01 * context_bias)
    #max pooling 
    #context_pool = batch * 1 * 1 * hidden_size
    context_pool = tf.nn.max_pool(
        context_cnn_bias,
        ksize=[1, length - conv_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name=scope_name+'_pool'
    )
    if reshape:
        return tf.reshape(context_pool, [-1, hidden_size]), con_W, con_b
    else:
        return context_pool, con_W, con_b

def CNNMereTensor(input, hidden_size, word_dim,
    conv_size, length, 
    scope_name = "Mere_CNN", reshape = True):
    filter_shape = [conv_size, word_dim, 1, hidden_size]
    con_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name=scope_name+'_W')
    con_b = tf.Variable(tf.constant(0.01, shape=[hidden_size]), name=scope_name+'_b')
    #batch * h * w * hidden, w == 1
    context_cnn = tf.nn.conv2d(
        tf.expand_dims(input, -1),
        con_W,
        strides=[1, 1, 1, 1],
        padding='VALID',
        name=scope_name+'_cnn'    
    )
    #context_cnn_bias = tf.nn.relu(tf.nn.bias_add(context_cnn, con_b), name= scope_name + 'relu')
    #context_cnn_bias = tf.nn.tanh(tf.nn.bias_add(context_cnn, con_b), name= scope_name + '_tanh')
    context_bias = tf.nn.bias_add(context_cnn, con_b)
    context_cnn_bias = tf.maximum(context_bias, -.01 * context_bias)
    reshape_context_cnn_bias = tf.reshape(context_cnn_bias, [-1, tf.shape(context_cnn_bias)[1], tf.shape(context_cnn_bias)[3]])

    if reshape:
        return reshape_context_cnn_bias, con_W, con_b
    else:
        return context_cnn_bias, con_W, con_b



def CNNReuseMereTensor(input, hidden_size, word_dim, 
    conv_size, length, 
    con_W, con_b, scope_name = 'Reuse_Mere_CNN', reshape = True):
    context_cnn = tf.nn.conv2d(
        tf.expand_dims(input, -1),
        con_W,
        strides = [1,1,1,1],
        padding = 'VALID',
        name = scope_name + '_cnn')
    # context_bias = tf.nn.bias_add(context_cnn, con_b)
    # context_cnn_bias = tf.maximum(context_bias, -.01 * context_bias)
    context_cnn_bias = tf.nn.relu(tf.nn.bias_add(context_cnn, con_b), name= scope_name + 'relu')
    reshape_context_cnn_bias = tf.reshape(context_cnn_bias, [-1, tf.shape(context_cnn_bias)[1], tf.shape(context_cnn_bias)[3]])

    if reshape:
        return reshape_context_cnn_bias
    else:
        return context_cnn_bias

def CNNReuseTensor(input, hidden_size, word_dim, 
    conv_size, length, 
    con_W, con_b, scope_name = 'Reuse_CNN'):
    context_cnn = tf.nn.conv2d(
        tf.expand_dims(input, -1),
        con_W,
        strides = [1,1,1,1],
        padding = 'VALID',
        name = scope_name + '_cnn')
    # context_bias = tf.nn.bias_add(context_cnn, con_b)
    # context_cnn_bias = tf.maximum(context_bias, -.01 * context_bias)
    context_cnn_bias = tf.nn.relu(tf.nn.bias_add(context_cnn, con_b), name= scope_name + 'relu')

    context_pool = tf.nn.max_pool(
    context_cnn_bias,
    ksize=[1, length - conv_size + 1, 1, 1],
    strides=[1, 1, 1, 1],
    padding='VALID',
    name=scope_name+'_pool'
    )

    return tf.reshape(context_pool, [-1, hidden_size])


def Kernel_Pooling(q_cnn, e_cnn, mask, mu, sigma, nkernel):
    '''
    :param q_cnn: input queries. [nquery, qlen, hidden_size]
    :param e_cnn: input documents. [nquery, dlen, hidden_size]
    :param mask: a binary mask. [nquery, qlen, dlen]
    :param mu: kernel mu values. [1, 1, nkernel]
    :param sigma: kernel sigma values. [1, 1, nkernel]
    :param nkernel: number of kernel
    :param hidden_size: number of convolutional window
    :param wordembd_dim: word vector dimension
    :param conv_size: convolutional window size
    :return: return the predicted score for each <query, document> in the batch 
    '''

    #feed to cnn to get a better representation
    #query cnn [nquery, max_q_len - conv_size + 1, hidden_size]
    # q_cnn, _, _ = CNNMereTensor(q_embed, hidden_size, word_dim, conv_size, max_q_len, scope_name = 'Kernel_Query_CNN')
    # #entity cnn [nquery, max_e_len - conv_size + 1, hidden_size]
    # e_cnn, _, _ = CNNMereTensor(e_embed, hidden_size, word_dim, conv_size, max_e_len, scope_name = 'Kernel_Entity_CNN')
    #normalize and compute similarity matrix
    norm_q = tf.sqrt(tf.reduce_sum(tf.square(q_cnn), 2, keep_dims = True))
    norm_e = tf.sqrt(tf.reduce_sum(tf.square(e_cnn), 2, keep_dims = True))
    normalized_q_cnn = q_cnn / norm_q
    normalized_e_cnn = e_cnn / norm_e
    # normalized_q_cnn = q_cnn
    # normalized_e_cnn = e_cnn
    # tf.add_to_collection("normalize",tf.reduce_sum(normalized_q_cnn))
    # tf.add_to_collection("normalize",tf.reduce_sum(normalized_q_cnn))
    tmp = tf.transpose(normalized_e_cnn, perm = [0, 2, 1])

    #cosine similarity[nquery, qlen, dlen]
    sim  = tf.matmul(normalized_q_cnn, tmp, name = "cosine_similarity")

    #compute gaussin kernel
    #reshape_sim = tf.reshape(sim, [nquery, max_q_len, max_e_len, 1])
    reshape_sim = tf.expand_dims(sim, -1)
    tmp = tf.exp(-tf.square(tf.subtract(reshape_sim, mu)) / (tf.multiply(tf.square(sigma), 2)))
    #mask non-existing words
    tmp = tf.multiply( tmp, tf.cast(tf.expand_dims(mask, -1), tf.float32) )

    #feats = [] #store soft-TF feature from each paramater pair
    #sum up gaussin scores
    kde = tf.reduce_sum(tmp, [2])

    ########
    small_zeros_matrix = tf.fill(tf.shape(kde), 1e-7)
    kde = tf.where(tf.less_equal(kde, small_zeros_matrix), small_zeros_matrix, kde)
    kde_sqrt = tf.sqrt(kde)
    ########
    #kde = [nquery, qlen, nkernel]
    #kde_log = tf.log(tf.maximum(kde, 1e-10)) * 0.0001 # 0.01 used to scale down the data
    #kde = tf.sqrt(tf.maximum(kde, 1e-10))
    #aggregate_kde = [nquery, nkernel]
    aggregate_kde = tf.reduce_sum(kde_sqrt, [1])
    matching_score = tf.layers.dense(aggregate_kde, units = 1, activation = None, name = 'kernel_combine')
    matching_score = tf.reshape(matching_score, [-1])

    # feats.append(aggregate_kde)
    # kernel_W = tf.get_variable("Kernel_W", [nkernel, 1], initializer = tf.contrib.layers.xavier_initializer())
    # kernel_b = tf.Variable(tf.zeros([1]), name = "Kernel_b")
    # feats_tmp = tf.concat(feats, 1)
    # feats_flat = tf.reshape(feats_tmp, [-1, nkernel])

    # matching_score = tf.tanh(tf.matmul(tf.cast(feats_flat, tf.float32), kernel_W) + kernel_b)
    # #matching_score = tf.matmul(tf.cast(feats_flat, tf.float32), kernel_W) + kernel_b
    # matching_score = tf.reshape(matching_score, [-1])
    return matching_score


def MentionMLP(q_cnn, q_len, m_mask, hidden_size, scope_name = "mention_MLP"):
    '''
    :param q_cnn: input queries. [nquery, qlen, hidden_size]
    :param q_len: 
    :param m_mask: mask of mentions in queries [nquery]
    :param hidden_size: number of convolutional window
    :return: return the MLP representation of mention [nquery, hidden_size]
    '''	
    m_mask = tf.cast(m_mask, tf.float32)
    mask_q_cnn = q_cnn * tf.expand_dims(m_mask, -1)

    w1 = tf.Variable(tf.random_normal([hidden_size, hidden_size]), name = scope_name + '_W1')
    b = tf.Variable(tf.constant(0.0, shape = [hidden_size]), name = scope_name + '_b')

    reshape_mask_q_cnn = tf.reshape(mask_q_cnn, [-1, hidden_size])
    org_mention_mlp = tf.add(tf.matmul(reshape_mask_q_cnn, w1) + b)
    reshape_mention_mlp = tf.reshape(org_mention_mlp, [-1, q_len, hidden_size])

    mention_mlp = tf.reduce_sum(reshape_mention_mlp, 1)

    return mention_mlp

def MentionAttention(q_cnn, mention_mlp, q_mask, m_mask, q_len, hidden_size, scope_name = 'Mention_Attention'):
    '''
    :param q_cnn: input queries. [nquery, qlen, hidden_size]
   	:param mention_mlp: [nquery, hidden_size]
    :param q_mask: [nquery, qlen] 
    :param m_mask: mask of mentions in queries [nquery, q_len]
    :param hidden_size: number of convolutional window
    :return: return the context_cnn with attention score [nquery, qlen, hidden_size]
    		 att_score: attention score 
    '''
    m_mask = tf.cast(m_mask, tf.float32)
    q_mask = tf.cast(q_mask, tf.float32)

    w1 = tf.Variable(tf.random_normal([hidden_size, hidden_size]), name = scope_name + '_W1')
    w2 = tf.Variable(tf.random_normal([hidden_size, hidden_size]), name = scope_name + '_W2')
    w3 = tf.Variable(tf.random_normal([hidden_size, 1]), name = scope_name + '_W3') 


    weight_mention_mlp = tf.matmul(mention_mlp, w1)
    reshape_weight_mention_mlp = tf.expand_dims(weight_mention_mlp, 1)

    q_cnn_cp = q_cnn

    weight_q_cnn = tf.matmul(tf.reshape(q_cnn, (-1, hidden_size), w2))
    reshape_weight_q_cnn = tf.reshape(weight_q_cnn, [-1, q_len, hidden_size])

    weight_mention_q_cnn = tf.matmul(tf.reshape(tf.tanh(reshape_weight_q_cnn + reshape_weight_mention_mlp), [-1, hidden_size]), w3)
    reshape_weight_mention_q_cnn = tf.reshape(weight_mention_q_cnn, [-1, q_len])

    mask = (1 - m_mask) * q_mask

    unnormal_att_score = tf.nn.softmax(reshape_weight_mention_q_cnn, dim = 1)
    att_score = unnormal_att_score * mask / tf.reduce_sum(unnormal_att_score * mask)

    att_score += m_mask

    att_q_cnn = tf.expand_dims(att_score, -1) * q_cnn_cp

    return att_q_cnn, att_score

# def Matching_CNN_Embedding(q_cnn, e_cnn, config):
#     # 1 is the width
#     #q_cnn = [b * hq * 1 * hidden]
#     #e_cnn = [b * he * 1 * hidden]
#     q_cnn = tf.transpose(q_cnn, [1,2,0,3])
#     e_cnn = tf.transpose(e_cnn, [1,2,0,3])
#     #q_cnn e_cnn = [h * 1 * b * hidden]
#     # hq, wq, bq, hidden_q = tf.unstack(tf.shape(q_cnn))
#     # he, we, be, hidden_e = tf.unstack(tf.shape(e_cnn))

#     # q_cnn = tf.reshape(q_cnn, (hq, wq, bq * hidden_q, 1))
#     # e_cnn = tf.reshape(e_cnn, (1, he, we, be * hidden_e))
#     q_cnn = tf.reshape(q_cnn, shape = [config.local_ctx_len - config.conv_size + 1, 1, -1, 1])
#     e_cnn = tf.reshape(e_cnn, shape = [1, config.wiki_doc_len - config.conv_size + 1, 1, -1])

#     #cnn_final = [1 * hout * wout * ((b * hidden) * 1)] = [1 * hout * wout * (hidden * b)]
#     cnn_final = tf.nn.depthwise_conv2d(e_cnn, q_cnn, strides = [1,1,1,1], padding = "VALID")
#     # #cnn_final = [b * hout * wout * ((hidden) * 1)]
#     # cnn_final = tf.concat(tf.split(cnn_final, bq, axis = 3), axis = 0)
#     # #cnn_final = [b * hout * wout]
#     # cnn_final = tf.reduce_sum(cnn_final, axis = 3)
#     # cnn_final = tf.reduce_max(tf.reduce_max(cnn_final, axis = 1), axis = 1)
#     # cnn_final = tf.reshape(cnn_final, [-1])
#     _, hout, wout, _ = tf.unstack(tf.shape(cnn_final))
#     #cnn_final = 1 * hout * wout * b * hidden
#     cnn_final = tf.reshape(cnn_final, [1, hout, wout, -1 ,config.hidden_size])
#     cnn_final = tf.reduce_sum(cnn_final, axis = 4)
#     #hout * wout * b
#     cnn_final = tf.squeeze(cnn_final, axis = 0)
#     #b * hout * wout
#     cnn_final = tf.transpose(cnn_final, [2, 0, 1])
#     #get the max number of each pair
#     cnn_final = tf.reduce_max(tf.reduce_max(cnn_final, axis = 2), axis = 1)
#     cnn_final = tf.reshape(cnn_final, [-1])
#     return cnn_final

def Matching_CNN_Embedding(q_cnn, e_cnn, config):
    # 1 is the width
    #q_cnn = [b * hq * 1 * hidden]
    #e_cnn = [b * he * 1 * hidden]
    q_cnn = tf.transpose(q_cnn, [1,2,0,3])
    e_cnn = tf.transpose(e_cnn, [1,2,0,3])
    #q_cnn e_cnn = [h * 1 * b * hidden]
    hq, wq, bq, hidden_q = tf.unstack(tf.shape(q_cnn))
    he, we, be, hidden_e = tf.unstack(tf.shape(e_cnn))

    q_cnn = tf.reshape(q_cnn, shape = [hq, wq, bq * hidden_q, 1])
    e_cnn = tf.reshape(e_cnn, shape = [1, he, we, be * hidden_e])
    # q_cnn = tf.reshape(q_cnn, shape = [config.local_ctx_len - config.conv_size + 1, 1, -1, 1])
    # e_cnn = tf.reshape(e_cnn, shape = [1, config.wiki_doc_len - config.conv_size + 1, 1, -1])

    #cnn_final = [1 * hout * wout * ((b * hidden) * 1)] = [1 * hout * wout * (hidden * b)]
    cnn_final = tf.nn.depthwise_conv2d(e_cnn, q_cnn, strides = [1,1,1,1], padding = "VALID")
    # #cnn_final = [b * hout * wout * ((hidden) * 1)]
    # cnn_final = tf.concat(tf.split(cnn_final, bq, axis = 3), axis = 0)
    # #cnn_final = [b * hout * wout]
    # cnn_final = tf.reduce_sum(cnn_final, axis = 3)
    # cnn_final = tf.reduce_max(tf.reduce_max(cnn_final, axis = 1), axis = 1)
    # cnn_final = tf.reshape(cnn_final, [-1])
    _, hout, wout, _ = tf.unstack(tf.shape(cnn_final))
    #cnn_final = 1 * hout * wout * b * hidden
    cnn_final = tf.reshape(cnn_final, [1, hout, wout, -1 ,config.hidden_size])
    cnn_final = tf.reduce_sum(cnn_final, axis = 4)
    #hout * wout * b
    cnn_final = tf.squeeze(cnn_final, axis = 0)
    #b * hout * wout
    cnn_final = tf.transpose(cnn_final, [2, 0, 1])
    #get the max number of each pair
    cnn_final = tf.reduce_max(tf.reduce_max(cnn_final, axis = 2), axis = 1)
    cnn_final = tf.reshape(cnn_final, [-1])
    return cnn_final