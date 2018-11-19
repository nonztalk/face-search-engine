#!/usr/bin/env python3
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.python.framework import meta_graph

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


def main (_):
    model = Model()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.loader(sess)
        for image_path in glob(IMAGE_PATH):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            batch = cv2.resize(image, (IMAGE_SZ, IMAGE_SZ)) 
            batch = batch[np.newaxis, :, :, :]
            embeddings = sess.run(model.embeddings, feed_dict={model.images: batch})
            print(embeddings.shape)
    pass

if __name__ == '__main__':
    tf.app.run()

