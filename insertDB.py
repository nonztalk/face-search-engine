#!/usr/bin/env python3

import os
import fawn
import face_detect_vectorize as fdv


# extract features and messages of photos in the picture database
def extract_features(barcode):
    faces = {}
    figures = fdv.read_pickle(IMAGE_OUTPUT_PATH + barcode, "info")
    if len(figures) == 0:
        return faces
    else:
        for figure in figures:
            faces[figure['ID']] = figure['vector']
        return faces


IMAGE_SZ = 160
IMAGE_OUTPUT_PATH = "./image_output/"

# The server url of image database
client = fawn.Fawn('http://127.0.0.1:8000')

# insert the image information into the database
# use the meta (another url) to reach the images
barcodes = [b for b in os.listdir(IMAGE_OUTPUT_PATH) if b.isdigit()]
for barcode in barcodes:
    faceFeatures = extract_features(barcode=barcode)
    if faceFeatures == {}:
        print('Empty data, continue ... ')
        continue
    else:
        for faceId, faceVector in faceFeatures.items():
            meta = {'barcode': barcode}
            print("%s_%s: %s" % (barcode, faceId, faceVector.shape))
            client.insert('%s_%s' % (barcode, faceId), faceVector.reshape(512), meta)
client.sync()
