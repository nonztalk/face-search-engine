#!/usr/bin/env python3
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tvm
import nnvm
import nnvm.testing.tf

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import meta_graph

IMAGE_SZ = 160
MODEL = '20180402-114759/20180402-114759.pb'

target = 'llvm'
target_host = 'llvm'
layout = None
ctx = tvm.cpu(0)

images = tf.placeholder(tf.uint8, shape=(None, IMAGE_SZ, IMAGE_SZ, 3), name="images")
batch = (tf.cast(images, tf.float32) - 127.5) / 128.0
is_training = tf.constant(False)

with tf.gfile.FastGFile(MODEL, 'rb') as PB:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(PB.read())
    tf.import_graph_def(graph_def, name = 'FaceNet', 
            input_map = {'image_batch:0': batch, 'phase_train': is_training},
            return_elements = ['embeddings:0'])
    graph_def = tf.graph_util.remove_training_nodes(graph_def)
    graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)
    sym, params = nnvm.frontend.from_tensorflow(graph_def, layout=layout)


