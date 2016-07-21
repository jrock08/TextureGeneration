import sys
sys.path.insert(0, 'pyAIUtils')

import features
import models
import os.path

import gflags
import math
import numpy as np
import random
import tensorflow as tf

from scipy.optimize import fmin_l_bfgs_b

import aiutils.vis.image_io as image_io


gflags.DEFINE_string('input_image', None, '')
gflags.DEFINE_string('output_image', None, '')
gflags.DEFINE_list('output_image_shape', ['100','100'], '')


def generate_image(input_image, shape):
    g = tf.Graph()
    with g.as_default():
        inp = tf.constant((np.expand_dims(input_image,0)/256.0)*2.0 - 1.0, dtype=tf.float32)
        with tf.variable_scope('feature'):
            inp_model = models.Feature(inp)
            inp_feature = inp_model.y()

        filter_half_size = 56/2
        shape_full = (shape[0], shape[1]+2*filter_half_size, shape[2]+2*filter_half_size, shape[3])

        inp_im_flat = tf.placeholder(tf.float32, np.prod(shape_full))
        out_im = tf.Variable(np.random.random(shape_full),dtype=tf.float32)
        with tf.variable_scope('feature', reuse=True):
            with tf.control_dependencies([out_im.assign(tf.reshape(inp_im_flat, shape_full))]):
                out_im_model = models.Feature(out_im)
                out_im_feature = out_im_model.y()

        inp_shape = inp_feature.get_shape().as_list()
        gram_inp = features.gram_matrix(inp_feature[:,filter_half_size:inp_shape[1]-filter_half_size, filter_half_size:inp_shape[2]-filter_half_size, :])

        out_shape = out_im_feature.get_shape().as_list()
        gram_out = features.gram_matrix(out_im_feature[:,filter_half_size:out_shape[1]-filter_half_size, filter_half_size:out_shape[2]-filter_half_size, :])

        gram_loss = 1.0/tf.reduce_sum(gram_inp**2) * tf.reduce_sum((gram_inp-gram_out)**2)

        gram_weight = tf.Variable(10**7, dtype=tf.float32)
        full_loss = gram_weight*gram_loss
        grad = tf.gradients([full_loss], [out_im])

        sess = tf.Session()

        def func_w_grad(x0):
            f,g = sess.run([full_loss, grad[0]], {inp_im_flat:x0})
            return f, np.float64(np.ndarray.flatten(g))

        tf.initialize_all_variables().run(session=sess)

        initial_x = np.random.random(np.prod(shape_full))
        bounds = np.transpose(np.array([-np.ones(np.prod(shape_full)), np.ones(np.prod(shape_full))]))

        output = fmin_l_bfgs_b(func_w_grad, initial_x, bounds=bounds, maxiter=2000, iprint=50)

        return np.squeeze(np.reshape(output[0]/2+.5, shape_full))[filter_half_size:-filter_half_size, filter_half_size:-filter_half_size,:]



def main(argv):
    gflags.MarkFlagAsRequired('input_image')
    try:
        argv = gflags.FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], gflags.FLAGS)
        sys.exit(1)

    assert(gflags.FLAGS.input_image is not None)

    input_image = image_io.imread(gflags.FLAGS.input_image)
    assert(len(gflags.FLAGS.output_image_shape) == 2)
    shape = [1] + [int(x) for x in gflags.FLAGS.output_image_shape] + [3]

    im = generate_image(input_image, shape)
    out_im_file = gflags.FLAGS.output_image or os.path.splitext(gflags.FLAGS.input_image)[0] + '_out.png'
    image_io.imwrite(np.uint8(im*256), out_im_file)

if __name__ == '__main__':
    main(sys.argv)

