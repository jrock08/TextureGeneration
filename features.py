import tensorflow as tf

def gram_matrix(input):
    [n_img, h, w, nfeats] = input.get_shape().as_list()
    inp_split = tf.split(0, n_img, input)
    G_mat = []
    for inp in inp_split:
        feat = tf.reshape(inp, [-1, nfeats])
        gram = tf.matmul(tf.transpose(feat), feat)/(h*w*nfeats)
        G_mat.append(tf.reshape(gram, [1,-1]))

    return tf.concat(0,G_mat)
