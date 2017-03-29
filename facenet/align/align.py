from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
from facenet import facenet
from facenet.align import detect_face
import random
from time import sleep

def align(image_path):
  print('Creating networks and loading parameters')
    
  with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
      pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
  
  minsize = 20 # minimum size of face
  threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
  factor = 0.709 # scale factor

  # Add a random key to the filename to allow alignment using multiple processes
  random_key = np.random.randint(0, high=99999)
  filename = os.path.splitext(os.path.split(image_path)[1])[0]

  try:
    img = misc.imread(image_path)
  except (IOError, ValueError, IndexError) as e:
    errorMessage = '{}: {}'.format(image_path, e)
    print(errorMessage)
  else:
    if img.ndim < 2:
      print('Unable to align "%s"' % image_path)
      return

    if img.ndim == 2:
      img = facenet.to_rgb(img)
    img = img[:,:,0:3]

    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    print('nrof_faces: %s' % nrof_faces)
    n = 0
    if nrof_faces > 0:
      det = bounding_boxes[:,0:4]
      img_size = np.asarray(img.shape)[0:2]
      if nrof_faces >= 1:
        bounding_box_size = (det[:,2] - det[:,0]) * (det[:,3] - det[:,1])
        img_center = img_size / 2
        offsets = np.vstack([ (det[:,0] + det[:,2]) / 2 - img_center[1], (det[:,1] + det[:,3]) / 2 - img_center[0] ])
        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
        #index = np.argmax(bounding_box_size - offset_dist_squared * 2.0) # some extra weight on the centering
        #det = det[index,:]
        for one in det:
          one = np.squeeze(one)
          bb = np.zeros(4, dtype=np.int32)
          bb[0] = np.maximum(one[0] - 5.0 / 2, 0)
          bb[1] = np.maximum(one[1] - 5.0 / 2, 0)
          bb[2] = np.minimum(one[2] + 5.0 /2, img_size[1])
          bb[3] = np.minimum(one[3] + 5.0 /2, img_size[0])
          cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
          scaled = misc.imresize(cropped, (128, 128), interp='bilinear')
          misc.imsave('/dl/' + str(n) + '.png', scaled)
          n += 1
    else:
      print('Unable to align "%s"' % image_path)

if __name__ == "__main__":
  align(sys.argv[1])
