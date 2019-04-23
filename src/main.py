'''
The pipeline was developed in jupyter notebook.
This main.py is only a record for github reading.
It is not runnable for sure. Indents need adjustment.
'''
############Import Packages and Models###########
experiment=None
# experiment (optional)
from comet_ml import Experiment
experiment = Experiment(api_key="YkPEmantOag1R1VOJmXz11hmt", parse_args=False, project_name='ocr_pipeline')
experiment.set_name('pipeline_debug')


import numpy as np
import argparse
import time
import cv2
import os
import tensorflow as tf
from glob import glob
from os.path import join, basename, dirname
from east.model import *
from recognition.Model import Model, DecoderType
from recognition.utils import log_image


###########Define Public Arguments#################

parser = argparse.ArgumentParser()
#general:
parser.add_argument("-debug_folder", "--debug_folder", type=str,default='/root/yq/WordSegmentationRecognitionPipeline/src/debug/',help="path to debug folder")

# EAST model:
## parameter
parser.add_argument("-east_w", "--east_width", type=int, default=1920,help=" east model resized image width (should be multiple of 32)")
parser.add_argument("-east_e", "--east_height", type=int, default=1920,help="east model resized image height (should be multiple of 32)")
parser.add_argument("-mini_conf", "--mini_conf", type=int, default=0.5,help="mini_confidence for crop")

## ckpt
parser.add_argument("-east", "--east", type=str,default='/root/yq/WordSegmentationRecognitionPipeline/src/east/frozen_east_text_detection.pb',help="path to input EAST text detector")
## input image root
parser.add_argument("-image_root", "--image_root", type=str,default='/root/yq/WordSegmentationRecognitionPipeline/src/Inputs/',help="path to input image root")


# Recognition Model
# basic operations
parser.add_argument("-name", default='dense_128_32_noartifact_beamsearch_prt', type=str, help="name of the log")
parser.add_argument("-gpu", default='-1', type=str, help="gpu numbers")
#parser.add_argument("-train", help="train the NN", action="store_true")
#parser.add_argument("-validate", help="validate the NN", action="store_true")
parser.add_argument("-transfer", action="store_true")
#actually not effective:
parser.add_argument("-batchesTrained", default=0, type=int, help='number of batches already trained (for lr schedule)') 
# beam search
parser.add_argument("-beamsearch", help="use beam search instead of best path decoding",default=True, action="store_true")
parser.add_argument("-wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
# training hyperparam
parser.add_argument("-batchsize", default=50, type=int, help='batch size') # actually not effective in infrerence
parser.add_argument("-lrInit", default=1e-2, type=float, help='initial learning rate') # actually not effective
parser.add_argument("-optimizer", default='rmsprop', help="adam, rmsprop, momentum") # actually not effective
parser.add_argument("-wdec", default=1e-4, type=float, help='weight decay') # acctually not effective
#parser.add_argument("-lrDrop1", default=10, type=int, help='step to drop lr by 10 first time')
#parser.add_argument("-lrDrop2", default=1000, type=int, help='step to drop lr by 10 sexond time')
#parser.add_argument("-epochEnd", default=40, type=int, help='end after this many epochs')
# trainset hyperparam
#parser.add_argument("-noncustom", help="noncustom (original) augmentation technique", action="store_true")
#parser.add_argument("-noartifact", help="dont insert artifcats", action="store_true")
#parser.add_argument("-iam", help='use iam dataset', action='store_true')
# densenet hyperparam
parser.add_argument("-nondensenet", help="use noncustom (original) vanilla cnn", action="store_true")
parser.add_argument("-growth_rate", default=12, type=int, help='growth rate (k)')
parser.add_argument("-layers_per_block", default=18, type=int, help='number of layers per block')
parser.add_argument("-total_blocks", default=5, type=int, help='nuber of densenet blocks')
parser.add_argument("-keep_prob", default=1, type=float, help='keep probability in dropout')
parser.add_argument("-reduction", default=0.4, type=float, help='reduction factor in 1x1 conv in transition layers')
parser.add_argument("-bc_mode", default=True, type=bool, help="bottleneck and compresssion mode")
# rnn,  hyperparams
parser.add_argument("-rnndim", default=256, type=int, help='rnn dimenstionality') #256
parser.add_argument("-rnnsteps", default=32, type=int, help='number of desired time steps (image slices) to feed rnn')
# img size
parser.add_argument("-imgsize", default=[128,32], type=int, nargs='+') #qyk default 128,32
# testset crop
#parser.add_argument("-crop_r1", default=3, type=int)
#parser.add_argument("-crop_r2", default=28, type=int)
#parser.add_argument("-crop_c1", default=10, type=int)
#parser.add_argument("-crop_c2", default=115, type=int)
# filepaths
#parser.add_argument("-dataroot", default='/root/datasets', type=str)
parser.add_argument("-ckptroot", default='/root/ckpt', type=str)
#parser.add_argument("-urlTransferFrom", default=None, type=str)

args = parser.parse_known_args()[0]
home = os.environ['HOME']
name = args.name
ckptroot = join(home, 'ckpt')
args.ckptpath = join(ckptroot, name)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

###################Set Image Path################################
image_paths=glob(args.image_root+'**.jpg')

###################Initiate Models###############################
model_east=east_cv2(args,experiment=experiment)

decoderType = DecoderType.BestPath
if args.beamsearch:
    decoderType = DecoderType.BeamSearch
elif args.wordbeamsearch:
    decoderType = DecoderType.WordBeamSearch

model_recg = Model(args, open(join(args.ckptpath, 'charList.txt')).read(), decoderType, mustRestore=True)

#################CV2 EAST Detection##############################
images,boundingboxes=model_east.crop(image_paths[0])

#################Recognition#####################################
'''
Assumption:

The number of words is less than 500, let's try to predict them in one batch
The images in images list variable are all on word level.
'''
recognizeds=model_recg.inferBatch(images)
#log experiment
if experiment !=None:
    result_sets=zip(images,recognizeds)
    for idx, (image, label) in enumerate(result_sets):
        text = '['+str(idx)+']: '+label
        log_image(experiment, image, text, '', args.debug_folder, counter='', epoch='')
        #counter += 1 # previous batch.imgs[i]


