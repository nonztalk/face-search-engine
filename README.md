# face-search-engine
This project aims to construct a face search engine using the techologies of face detection, face feature extraction and information retrieval. We use [SeetaFaceEngine](https://github.com/seetaface/SeetaFaceEngine) and its python API [pyseeta](https://github.com/TuXiaokang/pyseeta) to detect and label the faces from a given image, and use [FaceNet](https://github.com/davidsandberg/facenet) to extract the features of each face. The search engine is built upon the [donkey framework](https://github.com/aaalgo/donkey), which receive an uploaded image file and return K most similar faces and their origin images from our dataset. 
## 0. Environment
### Operating system
```bash
Distributor ID:	Ubuntu
Description:	Ubuntu 18.04.1 LTS
Release:	18.04
Codename:	bionic
```
### pyseeta
Just follow the instruction of pyseeta installation: https://github.com/TuXiaokang/pyseeta
### FaceNet pre-trained model
The pre-trained model can be downloaded from the following link. The file name need to change and make sure the following file are included in this directory:
```bash
20180402-114759.pb
model-20180402-114759.index
model-20180402-114759.meta
model-20180402-114759.data-00000-of-00001
```
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|--------------|
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
## 1. Face collection
In order to construct the search engine, the very first thing is to collect the faces, and manage their information in a well-defined file structure. Here is directory tree in our project:  
Suppose the image is 00000000.png, and pyseeta detects 3 faces, the output should be:
```bash
|-- output_dir
    `-- 00000000
        |-- 0.png
        |-- 1.png
        |-- 2.png
        |-- info
        |-- label.png
        `-- origin.png
```
0.png, 1.png, 2.png are the three clipped faces from 00000000.png, and origin.png, label.png are the original image and image with bounding boxes. Info is a binary file that store the face information in a dictionary:
```python
{
  'ID': i,                          # face id
  'left': face.left,                # bounding box coordinates
  'right': face.right,
  'top': face.top,
  'bottom': face.bottom,
  'score': face.score,              # face score
  'vector': face_feature            # face extracted feature
}
```
To run the example, just simply run `python3 face_main.py`. The generated files are stored in `sample_output`.  
To construct your own collection, just set `ORIGINAL_IMAGE=glob("your/image/path")` and `IMAGE_OUTPUT_PATH = "your/output/dir"` in `face_main.py`
## 2. Construct search engine
The donkey framework have already compiled in this repository. To construct a new database for the search engine, one needs to run `bash reset.sh` to reset the database, and open server `./server` before insert the data.  
In `donkey.xml` we define the address and port of the server. When changing them in the XML file, the corresponding code in `fawn.py` should also change.  
To insert data, run `python3 insertDB.py` when the server is open. And `searchDB.py` is a sample code that show the example of searching similar faces. The similarity is defined as L2 distance. Here is a running example which finds the 5 most similar face (the data is not included in the sample_image):
```bash
[{'details': '', 'key': '38_0', 'meta': '', 'score': 0.2991769015789032}, 
 {'details': '', 'key': '18_0', 'meta': '', 'score': 0.5391111969947815}, 
 {'details': '', 'key': '32_0', 'meta': '', 'score': 0.5569632053375244}, 
 {'details': '', 'key': '189_0', 'meta': '', 'score': 0.5968195796012878}, 
 {'details': '', 'key': '683_0', 'meta': '', 'score': 0.6704939603805542}]
```
where "key" is a unique id of each record in the database and the "score" is the L2 distance between face features. "Key" is organized as "imageDir_imageFileName", thus one can use key to locate the target image. For example, use "38_0" we can find the first face in 38.png by `the/dir/you/store/images_output/38/0.png` and the corresponding original image by `the/dir/you/store/images_output/38/origin.png`.

        
             
