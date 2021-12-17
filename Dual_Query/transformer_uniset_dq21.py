# 使用attention结构作为预测函数
# 序列中包括片段、文本、segment、memory四种节点
# 输出时选择与query相关的部分

import tensorflow as tf
import numpy as np

D_MODEL = 1024
D_FF = 2048
LENGTH_MAX = 4000

def positional_encoding(inputs, positions, scope='positional_encoding'):
    E = D_MODEL
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(LENGTH_MAX)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, positions)
        return tf.to_float(outputs)

def segment_embedding(inputs, hp, scope='segment_embedding'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        visual_embs= tf.get_variable(name='clip_posemb', shape=[D_MODEL])
        visual_embs = tf.expand_dims(tf.expand_dims(visual_embs, 0), 0)  # 1*1*D_model
        segment_embs = tf.get_variable(name='segment_posemb', shape=[D_MODEL])
        segment_embs = tf.expand_dims(tf.expand_dims(segment_embs, 0), 0)  # 1*1*D_model
        query_embs = tf.get_variable(name='query_posemb', shape=[D_MODEL])
        query_embs = tf.expand_dims(tf.expand_dims(query_embs, 0), 0)  # 1*1*D_model
        memory_embs = tf.get_variable(name='memory_posemb', shape=[D_MODEL])
        memory_embs = tf.expand_dims(tf.expand_dims(memory_embs, 0), 0)  # 1*1*D_model

        visual_nodes, segment_nodes, query_nodes, memory_nodes = inputs

        visual_nodes += visual_embs
        segment_nodes += segment_embs
        query_nodes += query_embs
        memory_nodes += memory_embs

        outputs = tf.concat([visual_nodes, segment_nodes, query_nodes, memory_nodes], axis=1)
        return outputs

def mask(inputs, key_masks=None, type=None):
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)
        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")
    return outputs

def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs

def ln(inputs, epsilon=1e-8, scope="ln"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs

def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)
        outputs = tf.layers.dense(outputs, d_model, use_bias=False, activation=None)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs

def ff(inputs, num_units, dropout_rate, scope="positionwise_feedforward"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Normalize
        outputs = tf.layers.dropout(ln(outputs), dropout_rate)
        # outputs = tf.layers.dropout(outputs, dropout_rate)  # ln_new

        # Residual connection
        outputs += inputs
    return outputs

def encoder(input_nodes, src_masks, drop_out, training, hp):
    # 将输入节点编码，只输出visual节点对应的编码
    # 在最后一层加入预测concept的ffn
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        enc = input_nodes
        block_outputs = []
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                enc = multihead_attention(queries=enc,
                                             keys=enc,
                                             values=enc,
                                             key_masks=src_masks,
                                             num_heads=hp.num_heads,
                                             dropout_rate=drop_out,
                                             training=training,
                                             causality=False)
                enc = ff(enc, num_units=[D_FF, D_MODEL], dropout_rate=drop_out)
                block_outputs.append(enc)
        memory = enc

        return memory

def local_attention(frame_embs, hp):
    # 在每个shot内部，使用self-attention将输出的frame表征聚合为shot表征
    # frame_output: N*(seqlen*5)*D
    # shot_embs: N*seqlen*D

    # 添加CLS位
    d_model = frame_embs.get_shape().as_list()[-1]
    frame_embs = tf.concat(tf.split(frame_embs, hp.seq_len, axis=1), axis=0)  # (seqlen*N)*5*D
    cls_embs = tf.get_variable(name='cls_emb', shape=[hp.seq_len*hp.gpu_num*hp.bc, 1, d_model])  # (seqlen*N)*1*D
    frame_embs = tf.concat([cls_embs, frame_embs], axis=1)  # (seqlen*N)*6*D
    enc = frame_embs

    for i in range(hp.num_blocks_local):
        with tf.variable_scope("num_blocks_local_{}".format(i), reuse=tf.AUTO_REUSE):
            Q = tf.layers.dense(enc, d_model, use_bias=True)  # (seqlen*N)*6*D
            K = tf.layers.dense(enc, d_model, use_bias=True)
            V = tf.layers.dense(enc, d_model, use_bias=True)
            Q_ = tf.concat(tf.split(Q, hp.num_heads, axis=2), axis=0)  # (h*seqlen*N)*6*(D/h)，先按照head数目切分特征
            K_ = tf.concat(tf.split(K, hp.num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, hp.num_heads, axis=2), axis=0)

            # attention
            attention = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (seqlen*h*N)*6*6
            attention /= d_model ** 0.5
            attention = tf.nn.softmax(attention)
            outputs_ = tf.matmul(attention, V_)  # (h*seqlen*N)*6*(D/h)
            outputs = tf.concat(tf.split(outputs_, hp.num_heads, axis=0), axis=2)  # (seqlen*N)*6*D，还原各个head
            outputs = tf.layers.dense(outputs, d_model, use_bias=False, activation=None)
            outputs += enc
            outputs = ln(outputs)
            enc = ff(outputs, num_units=[D_FF, D_MODEL], dropout_rate=0)

    shot_embs = enc[:, 0:1, :]  # (seqlen*N)*1*D
    shot_embs = tf.concat(tf.split(shot_embs, hp.seq_len, axis=0), axis=1)  # N*seqlen*D
    return shot_embs

def transformer(img_embs, segment_embs, txt_embs, memory_embs,
                segment_positions, feature_positions, indexes,
                scores_src, drop_out, training, hp):
    with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
        # encoder & decoder inputs
        visual_nodes = tf.layers.dense(img_embs, D_MODEL, use_bias=True, activation=None)
        segment_nodes = tf.layers.dense(segment_embs, D_MODEL, use_bias=True, activation=None)
        query_nodes = tf.layers.dense(txt_embs, D_MODEL, use_bias=True, activation=None)
        memory_nodes = tf.layers.dense(memory_embs, D_MODEL, use_bias=True, activation=None)

        visual_nodes += positional_encoding(visual_nodes, feature_positions)
        segment_nodes += positional_encoding(segment_nodes, segment_positions)

        input_nodes = [visual_nodes, segment_nodes, query_nodes, memory_nodes]
        input_nodes = segment_embedding(input_nodes, hp)

        src_masks = tf.math.equal(scores_src, 0)  # 标记输入的节点序列内哪些是padding部分

        # encoding & decoding
        enc_output = encoder(input_nodes, src_masks, drop_out, training, hp)
        shot_output= enc_output[: , 0 : hp.seq_len, :]
        query_output = enc_output[:, -(hp.memory_num+hp.query_num) : -hp.memory_num, :]
        memory_output = enc_output[:, -hp.memory_num : ,]

        # visual branch
        visual_branch = tf.layers.dense(shot_output, 1024, use_bias=True, activation=tf.nn.relu)
        visual_branch = tf.layers.dense(visual_branch, 512, use_bias=True, activation=tf.nn.relu)
        visual_branch = tf.layers.dense(visual_branch, 256, use_bias=True, activation=None)

        # query branch
        query_branch = tf.layers.dense(query_output, 1024, use_bias=True, activation=tf.nn.relu)
        query_branch = tf.layers.dense(query_branch, 512, use_bias=True, activation=tf.nn.relu)
        query_branch = tf.layers.dense(query_branch, 256, use_bias=True, activation=None)  # bc*q_num*D

        # 引入视频片段与各个concept的相关性关系
        corr_mat = tf.matmul(visual_branch, tf.transpose(query_branch, [0, 2, 1]))  # bc*seqlen*q_num
        sigmoid_score = tf.sigmoid(corr_mat)  # 对每个query，计算所有片段的相关性，片段间不影响
        softmax_logits = tf.nn.softmax(corr_mat, axis=1)  # 对每个片段，计算其与所有query的相关性，首先对片段做归一化
        aux_score = tf.nn.softmax(tf.reduce_sum(softmax_logits, axis=2, keepdims=True), axis=1)  # 附加得分，对应到每个片段
        pred_matrix = (1 - hp.aux_pr) * sigmoid_score + hp.aux_pr * aux_score  # bc*seqlen*q_num

        # 选取与query相关的部分
        pred_scores = []
        for i in range(hp.bc):
            ind = indexes[i]
            pred_scores.append(pred_matrix[i : i + 1, :, ind])
        pred_scores = tf.concat(pred_scores, axis=0)  # bc*seqlen

        return shot_output, memory_output, pred_scores

