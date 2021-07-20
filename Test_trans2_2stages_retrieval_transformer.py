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
    # 每个序列中的clip各自对应一个嵌入，所有concept共享一个嵌入
    # 将inputs按照clip与concept切分，clip加上positional embedding，concept加上一个共享的embedding
    concept_len = inputs.get_shape().as_list()[1] - (hp.seq_len + 1)  # 多一个global节点
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        global_emb = tf.get_variable(name='global_posemb', shape=[D_MODEL])
        global_emb = tf.expand_dims(tf.expand_dims(global_emb, 0), 0)  # 1*1*D_model
        clip_embs= tf.get_variable(name='clip_posemb', shape=[D_MODEL])
        clip_embs = tf.expand_dims(tf.expand_dims(clip_embs, 0), 0)  # 1*1*D_model
        concept_embs = tf.get_variable(name='concept_posemb', shape=[D_MODEL])
        concept_embs = tf.expand_dims(tf.expand_dims(concept_embs, 0), 0)  # 1*1*D_model

        global_node, clip_nodes, concept_nodes = tf.split(inputs, [1, hp.seq_len, concept_len], 1)  # 沿seq_len切开

        global_node += global_emb
        clip_nodes += clip_embs
        concept_nodes += concept_embs

        outputs = tf.concat([global_node, clip_nodes, concept_nodes], axis=1)
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

def transformer(features, positions, scores_src, img_emb, global_emb, drop_out, training, hp, c_num, s_num):
    with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
        # encoder & decoder inputs
        image_nodes = tf.layers.dense(img_emb, D_MODEL, use_bias=True, activation=None)
        visual_nodes = tf.layers.dense(features, D_MODEL, use_bias=True, activation=None)
        global_nodes = tf.layers.dense(global_emb, D_MODEL, use_bias=True, activation=None)  # bc*1*D

        visual_nodes += positional_encoding(visual_nodes, positions)
        input_nodes = tf.concat([global_nodes, visual_nodes, image_nodes], axis=1)
        input_nodes = segment_embedding(input_nodes, hp)

        src_masks = tf.math.equal(scores_src, 0)  # 标记输入的节点序列内哪些是padding部分

        # encoding & decoding
        memory = encoder(input_nodes, src_masks, drop_out, training, hp)
        global_node, clip_nodes, concept_nodes = tf.split(memory, [1, hp.seq_len, c_num], 1)
        concept_nodes = tf.transpose(concept_nodes, perm=(0,2,1))  # bc*D*c
        decoder_output = tf.matmul(clip_nodes,concept_nodes)

        concept_branch = tf.layers.dense(decoder_output, 1024, use_bias=True, activation=tf.nn.relu)
        concept_branch = tf.layers.dense(concept_branch, 512, use_bias=True, activation=tf.nn.relu)
        concept_logits = tf.layers.dense(concept_branch, c_num, use_bias=True, activation=None)
        concept_logits = tf.nn.softmax(concept_logits, axis=1)  # 归一化

        summary_branch = tf.layers.dense(decoder_output, 1024, use_bias=True, activation=tf.nn.relu)
        summary_branch = tf.layers.dense(summary_branch, 512, use_bias=True, activation=tf.nn.relu)
        summary_logits = tf.layers.dense(summary_branch, s_num, use_bias=True, activation=None)
        summary_logits = tf.nn.softmax(summary_logits, axis=1)

        return concept_logits, summary_logits