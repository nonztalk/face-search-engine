import os
import sys
import tensorflow as tf
import face_detect_vectorize as fdv
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_PATH = '20180402-114759/model-20180402-114759'
IMAGE_SZ = 160
MIN_FACE_SIZE = 30
ORIGINAL_IMAGE = glob('./sample_image/*.*')
IMAGE_OUTPUT_PATH = "./sample_output/"

model = fdv.Model(path=MODEL_PATH, image_size=IMAGE_SZ)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print("Load faceNet =>>>")
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    model.loader(sess)
    print("Done!\n")

    for image in ORIGINAL_IMAGE:
        print('Processing %s ... ' % image)
        faces_info = []
        target_dir = IMAGE_OUTPUT_PATH + os.path.splitext(os.path.basename(image))[0]
        print("Save to %s" % target_dir)

        if not os.path.exists(target_dir):
            os.system('mkdir -p ' + target_dir)
            try:
                face = fdv.Face(image_path=image, output_dir=target_dir, min_face_size=MIN_FACE_SIZE)
                face.detect_face()
                face.write_origin()
                face.write_clip(write_to_file=True)
                face.write_label()
                face.vectorize(sess=sess, model=model, model_input_size=(IMAGE_SZ, IMAGE_SZ))
                face.write_info("info")
            except:
                os.system('mv %s %s_fail' % (target_dir, target_dir))
                sys.exit()
