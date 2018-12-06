
#! usr/bin/env python3
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
from datetime import timedelta
from pyseeta import Detector
from pyseeta import Aligner
from pyseeta import Identifier
import numpy as np
import tensorflow as tf
import fawn
import face_detect_vectorize as fdv
 
# allow file formats
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def test_detector(basepath, path):
    print('test detector:')
    # load model
    # print(basepath)

    detector = Detector()

    detector.set_min_face_size(30)

    image_color = cv2.imread(path, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    faces = detector.detect(image_gray)
    face_ls = []

    for i, face in enumerate(faces):
        tmp = image_color[face.top:face.bottom, face.left:face.right]
        cv2.imwrite(basepath+str(i)+".png", tmp)
        face_ls.append(str(i)+".png")

        #print('({0},{1},{2},{3}) score={4}'.format(face.left, face.top, face.right, face.bottom, face.score))
        #cv2.rectangle(image_color, (face.left, face.top), (face.right, face.bottom), (0,255,0), thickness=2)
        #cv2.putText(image_color, str(i), (face.left, face.bottom),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), thickness=1)
    #cv2.imshow('test', image_color)
    #cv2.waitKey(0)

    detector.release()

    return face_ls


# app
app = Flask(__name__)

# Global variables
IMAGE_SZ = 160
MODEL_PATH = '/ssd/shaochwu/project_650/20180402-114759/model-20180402-114759'
client = fawn.Fawn('http://127.0.0.1:8000')
  
model = fdv.Model(path=MODEL_PATH, image_size=IMAGE_SZ)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print("Load faceNet =>>>")
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
model.loader(sess)
print("Done!\n")

# cache time
app.send_file_max_age_default = timedelta(seconds=1)
 

@app.route('/', methods=['GET', 'POST']) 
def index():
    if request.method == 'POST':
        return redirect(url_for('upload'))

    return render_template('index.html')
 
# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET']) 
def upload():
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "Please check image format (valid format: png、PNG、jpg、JPG、bmp)"})
 
        #user_input = request.form.get("name")
 
        # current path
        basepath = os.path.dirname(__file__)  
 
        # remove existing images
        filelist = [ file for file in os.listdir(os.path.join(basepath, 'static/images'))]
        for file in filelist:
            os.remove(os.path.join(os.path.join(basepath, 'static/images'), file))

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  
        f.save(upload_path)
 
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
 
        return render_template('upload_ok.html')#, userinput=user_input)
 
    return render_template('upload.html')


@app.route('/upload/analyze', methods=['POST', 'GET']) 
def analyze():
    if request.method == 'POST':
        #f = request.form['image']
        #user_input = request.form.get("name")
 
        # current path
        basepath = os.path.dirname(__file__) 
        print(basepath) 
        path = os.path.join(basepath, 'static/images', 'test.jpg')
        face = fdv.Face(path, os.path.join(basepath, 'static/images/'), 30)
        face.detect_face()
        face.write_clip(write_to_file=True)
        face.vectorize(sess=sess, model=model, model_input_size=(IMAGE_SZ, IMAGE_SZ))
        
        input_faces = [str(i)+".png" for i in range(face.num_faces)]
        
        output_faces = {}
        origin_img = {}
        scores = {}
        count = 0
        for feature in face.face_features:
            tmp = client.search(feature.reshape(512), K=5)
            out = []
            org = []
            score = []
            for f in tmp:
                barcode = f['key'].split('_')
                score.append(round(f['score'],5))
                out.append(barcode[0]+'/'+barcode[1]+'.png')
                org.append(barcode[0]+'/origin.png')
            scores[input_faces[count]] = score 
            output_faces[input_faces[count]] = out
            origin_img[input_faces[count]] = org
            
            count += 1
        
        return render_template('analyze.html',input_faces=input_faces, output_faces=output_faces, output_scores=scores, origin_img=origin_img)#, userinput=user_input)
 
    return render_template('upload.html')
 
@app.route('/<path:path>')
def send_img(path):
    return send_from_directory('/ssd/dengkw/Project/image_output/',path)


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=8905, debug=True)

