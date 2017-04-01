# coding: utf-8

# input a dir and divide them into different folders of different persons
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import hashlib

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

import time
lastTime = time.time()
def cost(label):
    global lastTime
    print("%s cost %f s" % (label, time.time() - lastTime))
    lastTime = time.time()

def computeDist(emb, target_set):
    min_dist = 100

    for item in target_set:
        dist = np.sqrt(np.sum(np.square(np.subtract(emb, item))))
        if min_dist > dist:
            min_dist = dist

    return min_dist


class Watcher:
    def __init__(self, watch_path):
        cost('init graph and session')
        self.img_path = watch_path
        self.last_md5 = ''
        self.graph = tf.Graph()

        with self.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
            self.session = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options, log_device_placement = False))
            with self.session.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.session, None)
                cost('create mtcnn')

                # Load the model
                model_dir = '/dl/models/resnet'
                print('Model directory: %s' % model_dir)
                meta_file, ckpt_file = facenet.get_model_filenames(model_dir)
                print('Metagraph file: %s' % meta_file)
                print('Checkpoint file: %s' % ckpt_file)
                cost('start load model')
                facenet.load_model(model_dir, meta_file, ckpt_file)
                cost('load model done')

                # Get input and output tensors
                cost('start load vectors')
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                cost('loaded images_placeholder')
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                cost('loaded embeddings')
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                cost('loaded phase_train_placeholder')

                self.face_set = []

                
    def getFaces(self):
        minsize = 20 # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor
        
        faces = []
        img = misc.imread(self.img_path)
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        for box in bounding_boxes:
            det = np.squeeze(box[0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - 4, 0)
            bb[1] = np.maximum(det[1] - 4, 0)
            bb[2] = np.minimum(det[2] + 4, img_size[1])
            bb[3] = np.minimum(det[3] + 4, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (160, 160), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            faces.append(prewhitened)
        return faces


    def cluster(self):
        # iterate the target directory
        cost('start detect faces')
        faces = self.getFaces() 
        cost('detect faces[count = %d]' % len(faces))
        # Run forward pass to calculate embeddings
        feed_dict = { self.images_placeholder: faces, self.phase_train_placeholder: False }
        emb = self.session.run(self.embeddings, feed_dict = feed_dict)[0]
        cost('computing embedding')
        
        if len(faces) > 0:
            # found faces
            min_dist = 100
            min_index = -1
            for i in range(len(self.face_set)):
                items = self.face_set[i]
                dist = computeDist(emb, items)
                if dist < 1:
                    # same person
                    if min_dist > dist:
                        min_index = i
                        min_dist = dist

            if min_index > -1:
                print('%s found a familar face: %d, min_dist is %f' % (self.img_path, min_index, min_dist))
                self.face_set[min_index].append(emb)
            else:
                print('%s found a new face, min_dist is %f' % (self.img_path, min_dist))
                self.face_set.append([ emb ])
        else:
            print('no face')

    def start(self):
        while True:
            md5_new = md5(self.img_path)
            if os.path.exists(self.img_path) and md5_new != self.last_md5:
                print('changes')
                self.cluster()
            self.last_md5 = md5_new
            print(".",)
            time.sleep(1)
        
if __name__ == "__main__":
  import sys
  w = Watcher(sys.argv[1])
  w.start()
