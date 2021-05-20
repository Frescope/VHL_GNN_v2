# modified transformer for video_translate
import tensorflow as tf
import numpy as np

D_MODEL = 1024
D_FF = 2048

def positional_encoding(inputs, seq_len, scope='positional_encoding'):
    E = D_MODEL
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(seq_len)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        return tf.to_float(outputs)

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

        # block_outputs = tf.concat(block_outputs, axis=0)
        # [bc,seq,d] = memory.get_shape().as_list()
        # block_outputs = tf.reshape(block_outputs, shape=[hp.num_blocks, bc, seq, d])
        # block_outputs = tf.transpose(block_outputs, perm=[1,2,0,3])  # bc*seq*blocknum*d
        # block_outputs = tf.reshape(block_outputs, shape=[bc*seq, hp.num_blocks, d])  # (bc*seq)*blocknum*d
        # block_mask = tf.math.equal(tf.convert_to_tensor(np.ones([bc*seq, hp.num_blocks])), 0)
        # block_agg = multihead_attention(queries=block_outputs,
        #                                 keys=block_outputs,
        #                                 values=block_outputs,
        #                                 key_masks=block_mask,
        #                                 num_heads=4,
        #                                 dropout_rate=0,
        #                                 training=training,
        #                                 causality=False)
        # block_agg = ff(block_agg, num_units=[D_FF, D_MODEL], dropout_rate=0)
        # block_agg = tf.reduce_mean(block_agg, axis=1)  # (bc*seq)*d
        # block_agg = tf.reshape(block_agg, shape=[bc, seq, d])
        # memory = block_agg

        return memory

def decoder(decoder_input, memory, src_masks, tgt_masks, drop_out, training, hp):
    # 输入labels与encoder的输出，预测每个shot与各个concept的相关性，labels代表相关性的真值
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        dec = decoder_input
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # Masked self-attention (Note that causality is True at this time)
                dec = multihead_attention(queries=dec,
                                          keys=dec,
                                          values=dec,
                                          key_masks=tgt_masks,
                                          num_heads=hp.num_heads,
                                          dropout_rate=drop_out,
                                          training=training,
                                          causality=True,
                                          scope="decoder_self_attention")
                # encoder-decoder attention
                dec = multihead_attention(queries=dec,
                                          keys=memory,
                                          values=memory,
                                          key_masks=src_masks,
                                          num_heads=hp.num_heads,
                                          dropout_rate=drop_out,
                                          training=training,
                                          causality=False,
                                          scope="encoder_decoder_attention")
                dec = ff(dec, num_units=[D_FF, D_MODEL], dropout_rate=drop_out)
        return dec

def transformer(features, labels, scores_src, scores_tgt, txt_emb, img_emb, drop_out, training, hp):
    with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
        # encoder & decoder inputs
        image_nodes = tf.layers.dense(img_emb, D_MODEL, use_bias=True, activation=None)
        visual_nodes = tf.layers.dense(features, D_MODEL, use_bias=True, activation=None)
        # decoder_input = tf.layers.dense(labels, D_MODEL, use_bias=True, activation=None)

        visual_nodes += positional_encoding(visual_nodes, hp.seq_len)
        # decoder_input += positional_encoding(decoder_input, hp.seq_len)
        input_nodes = tf.concat([visual_nodes, image_nodes], axis=1)

        src_masks = tf.math.equal(scores_src, 0)  # 标记输入的节点序列内哪些是padding部分
        # tgt_masks = tf.math.equal(scores_tgt, 0)

        # encoding & decoding
        memory = encoder(input_nodes, src_masks, drop_out, training, hp)
        # decoder_output = decoder(decoder_input, memory, src_masks, tgt_masks, drop_out, training, hp)
        decoder_output = memory[:, :hp.seq_len, :]

        c_num = labels.get_shape().as_list()[-1]
        logits = tf.layers.dense(decoder_output, 512, use_bias=True, activation=tf.nn.relu)
        logits = tf.layers.dense(logits, 256, use_bias=True, activation=tf.nn.relu)
        logits = tf.layers.dense(logits, c_num, use_bias=True, activation=None)
        return logits


