#!/usr/bin/env python3
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.python.framework import meta_graph

import tvm
import nnvm
import nnvm.testing.tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

target = 'llvm'
target_host = 'llvm'
layout = None
ctx = tvm.cpu(0)

IMAGE_SZ = 160
MODEL_PATH = '20180402-114759/model-20180402-114759'
IMAGE_PATH = 'facenet/data/images/*.jpg'

class Model:
    def __init__ (self, path=MODEL_PATH, name='xxx'):
        images = tf.placeholder(tf.uint8, shape=(None, IMAGE_SZ, IMAGE_SZ, 3), name="images")
        batch = (tf.cast(images, tf.float32) - 127.5) / 128.0
        self.images = images
        is_training = tf.constant(False)
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        self.embeddings, = \
                tf.import_graph_def(mg.graph_def, name=name,
                        input_map={'image_batch:0': batch, 'phase_train': is_training},
                        return_elements=['embeddings:0'])
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)

        '''
        graph = tf.get_default_graph()
        graph.finalize()
        graph_def = graph.as_graph_def()
        for node in graph_def.node:
            print(node.name)
            pass
        sys.exit(0)
        '''
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

'''
flags.DEFINE_integer('clip_stride', 16, '')
flags.DEFINE_integer('max_size', 2000, '')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_string('colorspace', 'RGB', '')

flags.DEFINE_string('list', 'list', '')
flags.DEFINE_integer('max', 50, '')
'''

model = Model()
with tf.Session() as sess:
    model.loader(sess)
    graph_def = tf.graph_util.remove_training_nodes(sess.graph_def)
    graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)
    sym, params = nnvm.frontend.from_tensorflow(graph_def, layout=layout)

