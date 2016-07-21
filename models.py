import tensorflow as tf
import numpy as np
import math

import aiutils.tftools.images as images
import aiutils.tftools.layers as layers
from network import Network

class Feature(Network):
    def __init__(self, input):
        super(Feature, self).__init__(input)

    def setup(self):
        inp = self.get_output()
        self.conv(128, filter_size=1, name='conv1', inp_layer=inp)
        self.conv(128, filter_size=3, name='conv3', inp_layer=inp)
        self.conv(128, filter_size=5, name='conv5', inp_layer=inp)
        self.conv(128, filter_size=7, name='conv7', inp_layer=inp)
        self.conv(128, filter_size=11, name='conv11', inp_layer=inp)
        self.conv(128, filter_size=23, name='conv23', inp_layer=inp)
        self.conv(128, filter_size=37, name='conv37', inp_layer=inp)
        self.conv(128, filter_size=55, name='conv55', inp_layer=inp)

        concat_layer = tf.concat(3, [self.layerdict['conv1'], self.layerdict['conv3'], self.layerdict['conv5'],
                self.layerdict['conv7'], self.layerdict['conv11'],
                self.layerdict['conv23'], self.layerdict['conv37'],
                self.layerdict['conv55']])

        self.add_('concat', concat_layer)

    def y(self):
        return self.get_output()
