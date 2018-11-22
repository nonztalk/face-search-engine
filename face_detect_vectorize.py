import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow.python.framework import meta_graph
from pyseeta import Detector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model paths and settings
MODEL_PATH = '20180402-114759/model-20180402-114759'
IMAGE_SZ = 160

# pyseeta setting
MIN_FACE_SIZE = 20


class Model:
    def __init__(self, path=MODEL_PATH, name='xxx'):
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


def detect_clip_face(origin_image, target_dir):
    detector = Detector()
    detector.set_min_face_size(MIN_FACE_SIZE)

    # detect faces
    image_color = cv2.imread(origin_image, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    faces = detector.detect(image_gray)

    # clip and store faces
    cv2.imwrite(os.path.join(target_dir, "origin.png"), image_color)  # store the original picture
    clipped_faces = []
    for i, face in enumerate(faces):
        # store the clipped faces
        clipped_face = image_color[face.top:face.bottom, face.left:face.right]
        clipped_faces.append(clipped_face)
        cv2.imwrite(os.path.join(target_dir, str(i) + ".png"), clipped_face)
        # store the labeled faces
        cv2.rectangle(image_color, (face.left, face.top), (face.right, face.bottom), (0, 255, 0), thickness=2)
        cv2.putText(image_color, str(i), (face.left, face.bottom), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                    thickness=1)
        cv2.imwrite(os.path.join(target_dir, "labeled.png"), image_color)

    detector.release()

    return faces, clipped_faces


def face_vector(faces):
    face_vectors = []
    for i, face in enumerate(faces, start=1):
        print("Vectoring the %d/%d face ... " % (i, len(faces)))
        batch = cv2.resize(face, (IMAGE_SZ, IMAGE_SZ))
        batch = batch[np.newaxis, :, :, :]
        embeddings = sess.run(model.embeddings, feed_dict={model.images: batch})
        print(embeddings.shape)
        face_vectors.append(embeddings)
    return face_vectors


def write_pickle(target_dir, filename, obj):
    pickle.dump(obj, open(os.path.join(target_dir, filename), 'wb'))


def read_pickle(target_dir, filename):
    return pickle.load(open(os.path.join(target_dir, filename), 'rb'))


if __name__ == '__main__':
    model = Model()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print("Load faceNet =>>>")
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.loader(sess)
        print("Done!\n")

        images = glob('./sample_image/*.*')
        for image in images:
            print('Processing %s ... \n' % image)
            faces_info = []
            target_dir = "./sample_output/" + os.path.splitext(os.path.basename(image))[0]
            os.system('mkdir -p ' + target_dir)

            print('Detecting and clipping faces =>>>')
            faces, clipped_faces = detect_clip_face(origin_image=image, target_dir=target_dir)
            print('%d faces detected\n' % len(clipped_faces))

            print('Vectoring clipped faces =>>>')
            face_vectors = face_vector(faces=clipped_faces)
            print("Done!\n")

            print('Writing the information file =>>>')
            for i, face in enumerate(faces):
                faces_info.append({
                    'ID': i,
                    'left': face.left,
                    'right': face.right,
                    'top': face.top,
                    'bottom': face.bottom,
                    'score': face.score,
                    'vector': face_vectors[i]
                })
            write_pickle(target_dir=target_dir, filename="info", obj=faces_info)
            print("Done!\n")