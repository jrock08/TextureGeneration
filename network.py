import tensorflow as tf
import numpy as np

import aiutils.tftools.images as images
import aiutils.tftools.layers as layers
import aiutils.tftools.batch_normalizer as batch_normalizer

class BaseNetwork(object):
    def __init__():
        pass

    """ Call this after setup, while in the correct scope """
    def accumulate_variables(self):
        scope_name = tf.get_variable_scope().name
        assert scope_name != '', 'You almost certainly wanted to be in a variable scope when you created this network'

        # Only care about trainable variables
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)

class LegacyNetwork(BaseNetwork):
    def __init__(self, input, params_path=None, trainable=False):
        self.trainable = trainable
        if params_path:
            self.params = np.load(params_path).item()
        self.layers = []
        self.layerdict = {}
        self.batch_size = int(input.get_shape()[0])
        self.add_('input', input)
        self.setup()
        self.accumulate_variables()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def get_unique_name_(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var):
        self.layers.append((name, var))
        self.layerdict[name] = var

    def get_output(self):
        return self.layers[-1][1]

    def conv(self, h, w, c_i, c_o, stride=1, name=None):
        name = name or self.get_unique_name_('conv')
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('W', initializer=tf.constant(self.params[name][0].astype(np.float32)), trainable=self.trainable, dtype=tf.float32)
            conv = tf.nn.conv2d(self.get_output(), weights, [1,stride,stride,1], padding='SAME')
            if len(self.params[name]) > 1:
                biases = tf.get_variable('b', initializer=tf.constant(self.params[name][1].astype(np.float32)), trainable=self.trainable, dtype=tf.float32)
                bias = tf.nn.bias_add(conv, biases)
                relu = tf.nn.relu(bias, name=scope.name)
            else:
                relu = tf.nn.relu(conv, name=scope.name)
            self.add_(name, relu)
        return self

    def pool(self, size=2, stride=2, name=None):
        name = name or self.get_unique_name_('pool')
        # pool = tf.nn.avg_pool(self.get_output(),
        pool = tf.nn.max_pool(self.get_output(),
                              ksize=[1, size, size, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME',
                              name=name)
        self.add_(name, pool)
        return self

class Network(BaseNetwork):
    def __init__(self, input, phase_train=None):
        self.layers = []
        self.layerdict = {}
        self.batch_size = int(input.get_shape()[0])
        self.add_('input', input)
        if phase_train is not None:
            self.phase_train = phase_train
        else:
            self.phase_train = tf.placeholder_with_default(True, (), 'PhaseTrain')
        self.setup()
        self.accumulate_variables()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def get_unique_name_(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var):
        self.layers.append((name, var))
        self.layerdict[name] = var

    def get_output(self):
        return self.layers[-1][1]

    def bnconv(self, out_dim, stride=1, filter_size=3, name=None, func=tf.nn.relu, inp_layer=None):
        name = name or self.get_unique_name_('conv')
        if inp_layer is None:
            inp_layer = self.get_output()

        new_layer = layers.conv2d(inp_layer, filter_size, out_dim, name, strides=[1,stride,stride,1], func=None)
        new_layer = layers.batch_norm(new_layer, self.phase_train, name=name+'bn')
        if func is not None:
            new_layer = func(new_layer)

        self.add_(name, new_layer)
        return self

    def conv(self, out_dim, stride=1, filter_size=3, name=None, func=tf.nn.relu, inp_layer=None):
        name = name or self.get_unique_name_('conv')
        if inp_layer is None:
            inp_layer = self.get_output()

        new_layer = layers.conv2d(inp_layer, filter_size, out_dim, name, strides=[1,stride,stride,1], func=func)
        self.add_(name, new_layer)
        return self

    def atrous_conv2d(self, out_dim, rate=[1], filter_size=3, name=None, func=tf.nn.relu, inp_layer=None):
        name = name or self.get_unique_name_('aconv')
        if inp_layer is None:
            inp_layer = self.get_output()

        new_layers = []
        for r in rate:
            new_layers.append(layers.atrous_conv2d(inp_layer, filter_size, out_dim, rate=rate, func=func))

        self.add_(name, tf.concat(3, new_layers))
        return self

    def reduce_mean_image(self, name=None, inp_layer=None):
        name = name or self.get_unique_name_('reduce')
        if inp_layer is None:
            inp_layer = self.get_output()
        new_layer = tf.reduce_mean(inp_layer, [1,2])
        self.add_(name, new_layer)
        return self

    def bn_fully_connected(self, out_size, name=None, func=tf.nn.relu, inp_layer=None):
        name = name or self.get_unique_name_('fc')
        if inp_layer is None:
            inp_layer = self.get_output()

        new_layer = layers.full(inp_layer, out_size, name, func=None)
        new_layer = layers.batch_norm(new_layer, self.phase_train, name=name+'bn')

        if func is not None:
            new_layer = func(new_layer)

        self.add_(name, new_layer)
        return self

    def fully_connected(self, out_size, name=None, func=tf.nn.relu, inp_layer=None):
        name = name or self.get_unique_name_('fc')
        if inp_layer is None:
            inp_layer = self.get_output()
        new_layer = layers.full(inp_layer, out_size, name, func=func)
        self.add_(name, new_layer)
        return self

    def softmax_layer(self, num_classes, name=None, inp_layer = None):
        name = name or self.get_unique_name_('softmax')
        if inp_layer is None:
            inp_layer = self.get_output()
        new_layer = layers.full(inp_layer, num_classes, name, func=tf.nn.softmax)
        self.add_(name, new_layer)
        return self

    def batch_norm(self, name=None, inp_layer=None):
        name = name or self.get_unique_name_('bn')
        if inp_layer is None:
            inp_layer = self.get_output()
        new_layer = layers.batch_norm(inp_layer, self.phase_train, name=name)
        self.add_(name, new_layer)
        return self

    def resize_like(self, target_layer, name=None, inp_layer=None):
        name = name or self.get_unique_name_('resize_like')
        if inp_layer is None:
            inp_layer = self.get_output()
        with tf.variable_scope(name):
            new_layer = images.resize_images_like(inp_layer, self.layerdict[target_layer])
        self.add_(name, new_layer)

    def atrous_residual(self, outer_dim, inner_dim, name=None, inp_layer=None, rates=[1,4]):
        name = name or self.get_unique_name_('residual')
        if inp_layer is None:
            inp_layer = self.get_output()

        with tf.variable_scope(name):
            if inp_layer.get_shape().as_list()[-1] != outer_dim:
                input = layers.conv2d(inp_layer, 3, outer_dim, name+'_rch_conv', func=None)
                input = layers.batch_norm(input, self.phase_train, name=name+'_bn')
                input = tf.maximum(.01*input, input)
            else:
                input = inp_layer
            with tf.variable_scope('atrous_conv'):
                atrous_layers = []
                for rate in rates:
                    atrous_layers.append(layers.atrous_conv2d(input, 5, outer_dim // len(rates), 'aconv_'+str(rate), rate=rate, func=None))
                conv = tf.concat(3, atrous_layers)
                bn = layers.batch_norm(conv, self.phase_train, name='bn1')
                leaky_relu = tf.maximum(.01*bn, bn)
            conv2 = layers.conv2d(leaky_relu, 3, outer_dim, 'conv_outer', func=None)
            #bn2 = layers.batch_norm(conv, self.phase_train, name='bn2')
            res = conv2 + input
            bn2 = layers.batch_norm(res, self.phase_train, name='bn2')
            out = tf.maximum(.01*bn2, bn2)
            self.add_(name, out)
        return self

    def residual(self, outer_dim, inner_dim, name=None, inp_layer=None):
        name = name or self.get_unique_name_('residual')
        if inp_layer is None:
            inp_layer = self.get_output()

        with tf.variable_scope(name):
            if inp_layer.get_shape().as_list()[-1] != outer_dim:
                input = layers.conv2d(inp_layer, 3, outer_dim, name+'_rch_conv', func=None)
                input = layers.batch_norm(input, self.phase_train, name=name+'_bn')
                input = tf.maximum(.01*input, input)
            else:
                input = self.get_output()

            conv = layers.conv2d(input, 5, inner_dim, 'conv_inner', func=None)
            bn = layers.batch_norm(conv, self.phase_train, name='bn1')
            leaky_relu = tf.maximum(.01*bn, bn)

            conv2 = layers.conv2d(leaky_relu, 3, outer_dim, 'conv_outer', func=None)
            #bn2 = layers.batch_norm(conv, self.phase_train, name='bn2')
            res = conv2 + input
            bn2 = layers.batch_norm(res, self.phase_train, name='bn2')
            out = tf.maximum(.01*bn2, bn2)
            self.add_(name, out)

    def dropout(self, keep_prob=.8, name=None, inp_layer=None):
        name = name or self.get_unique_name_('dropout')
        if inp_layer is None:
            inp_layer = self.get_output()
        self.add_(name, tf.cond(self.phase_train,
            lambda: tf.nn.dropout(inp_layer, keep_prob),
            lambda: inp_layer))




