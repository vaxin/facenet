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

def getFaces(image_path):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    faces = []
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options, log_device_placement = False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
        img = misc.imread(image_path)
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
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

def computeDist(emb, target_set):
    min_dist = 100

    for item in target_set:
        dist = np.sqrt(np.sum(np.square(np.subtract(emb, item))))
        if min_dist > dist:
            min_dist = dist

    return min_dist

def cluster(path):
    # read all images
    
    face_set = []

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            model_dir = '/dl/models/resnet'
            print('Model directory: %s' % model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(model_dir)
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(model_dir, meta_file, ckpt_file)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # iterate the target directory
            for (root, dirs, files) in os.walk(path):
                for file_path in files:
                    img_path = root + '/' + file_path
                    faces = getFaces(img_path) 
                    # Run forward pass to calculate embeddings
                    feed_dict = { images_placeholder: faces, phase_train_placeholder: False }
                    emb = sess.run(embeddings, feed_dict = feed_dict)[0]
                   
                    min_dist = 100
                    min_index = -1
                    for i in range(len(face_set)):
                        items = face_set[i]
                        dist = computeDist(emb, items)
                        if dist < 1:
                            # same person
                            if min_dist > dist:
                                min_index = i
                                min_dist = dist
                    if min_index > -1:
                        print('%s found a familar face: %d, min_dist is %f' % (img_path, min_index, min_dist))
                        face_set[min_index].append(emb)
                    else:
                        print('%s found a new face, min_dist is %f' % (img_path, min_dist))
                        face_set.append([ emb ])

if __name__ == "__main__":
  import sys
  cluster(sys.argv[1])
