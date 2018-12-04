import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from pyseeta import Detector


class Model:
    def __init__(self, path, image_size, name='xxx'):
        images = tf.placeholder(tf.uint8, shape=(None, image_size, image_size, 3), name="images")
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


class Face:
    def __init__(self, image_path, output_dir, min_face_size):
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.image_grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_size = self.image.shape  # height, width, channel
        self.output_dir = output_dir
        self.min_face_size = min_face_size

        self.faces = None
        self.faces_matrices = []
        self.num_faces = 0
        self.face_features = []
        self.face_info = []

    def detect_face(self):
        detector = Detector()
        detector.set_min_face_size(self.min_face_size)
        print('Detecting and clipping faces =>>>')
        self.faces = detector.detect(self.image_grey)
        self.num_faces = len(self.faces)
        print('%d faces detected' % self.num_faces)
        detector.release()

    def write_origin(self):
        cv2.imwrite(os.path.join(self.output_dir, 'origin.png'), self.image)
        print("Origin writing Success!")

    def write_clip(self, write_to_file = False):
        for i, face in enumerate(self.faces):
            if face.left < 0:
                face.left = 0
            if face.bottom < 0:
                face.bottom = 0
            if face.right > self.image_size[1]:
                face.right = self.image_size[1]
            if face.top > self.image_size[0]:
                face.top = self.image_size[0]
            clipped_face = self.image[face.top:face.bottom, face.left:face.right]
            self.faces_matrices.append(clipped_face)
            if write_to_file:
                cv2.imwrite(os.path.join(self.output_dir, str(i) + ".png"), clipped_face)
        print("Clip writing Success!")

    def write_label(self):
        for i, face in enumerate(self.faces):
            cv2.rectangle(self.image, (face.left, face.top), (face.right, face.bottom), (0, 255, 0),
                          thickness=2)
            cv2.putText(self.image, str(i), (face.left, face.bottom), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0),
                        thickness=1)
        cv2.imwrite(os.path.join(self.output_dir, "label.png"), self.image)
        print("Label writing Success!")

    def draw_bounding_box(self, face_index):
        assert face_index < self.num_faces, 'We only have %s faces' % self.num_faces
        face = [f for f in self.face_info if f['ID'] == eval(face_index)][0]
        cv2.rectangle(self.image, (face['left'], face['top']), (face['right'], face['bottom']), (0, 255, 0), thickness=2)
        cv2.putText(self.image, str(face_index), (face['left'], face['bottom']), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                    thickness=1)

    def vectorize(self, sess, model, model_input_size=(160, 160)):
        for i, face in enumerate(self.faces_matrices, start=1):
            print("Vectoring the %d/%d face ... " % (i, self.num_faces))
            batch = cv2.resize(face, model_input_size)
            batch = batch[np.newaxis, :, :, :]
            embeddings = sess.run(model.embeddings, feed_dict={model.images: batch})
            self.face_features.append(embeddings)

    def write_info(self, filename):
        print('Writing the information file =>>>')
        for i, face in enumerate(self.faces):
            self.face_info.append({
                'ID': i,
                'left': face.left,
                'right': face.right,
                'top': face.top,
                'bottom': face.bottom,
                'score': face.score,
                'vector': self.face_features[i]
            })
        write_pickle(target_dir=self.output_dir, filename=filename, obj=self.face_info)
        print("Done!\n")

    def load_info(self, filename):
        info = read_pickle(target_dir=self.output_dir, filename=filename)
        self.face_info = info
        self.num_faces = len(info)


def write_pickle(target_dir, filename, obj):
    pickle.dump(obj, open(os.path.join(target_dir, filename), 'wb'))


def read_pickle(target_dir, filename):
    return pickle.load(open(os.path.join(target_dir, filename), 'rb'))


# if __name__ == '__main__':
#     model = Model()
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#
#     print("Load faceNet =>>>")
#     with tf.Session(config=config) as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(tf.local_variables_initializer())
#         model.loader(sess)
#         print("Done!\n")
#
#         for image in ORIGINAL_IMAGE:
#             print('Processing %s ... \n' % image)
#             faces_info = []
#             target_dir = IMAGE_OUTPUT_PATH + os.path.splitext(os.path.basename(image))[0]
#             os.system('mkdir -p ' + target_dir)
#
#             face = Face(image_path=image, output_dir=target_dir, min_face_size=MIN_FACE_SIZE)
#             face.detect_face()
#             face.write_origin()
#             face.write_clip()
#             face.vectorize(sess=sess, model=model, model_input_size=(IMAGE_SZ, IMAGE_SZ))
#             face.write_info("info")
