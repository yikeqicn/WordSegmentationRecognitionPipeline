import os
import cv2
from WordSegmentation import wordSegmentation, prepareImg


#from comet_ml import Experiment
#experiment = Experiment(api_key="YkPEmantOag1R1VOJmXz11hmt", parse_args=False, project_name='htr')
# yike: changed to my comet for debug
import sys
import argparse
import cv2
import editdistance
import numpy as np
import PIL
#from datasets import EyDigitStrings, IAM, IRS, PRT,PRT_WORD
#from datasets import PRT # use this to load data for debug. In practice, optional. You may use anything to load detected line images in batch.
from torch.utils.data import DataLoader, ConcatDataset, random_split#, SequentialSampler #yike: add SequentialSampler
import torchvision
import torchvision.transforms as transforms
# from DataLoader import DataLoader, Batch
# from DataLoaderMnistSeq import DataLoader, Batch
from Model import Model, DecoderType
#from SamplePreprocessor import preprocess
import os
from os.path import join, basename, dirname
import matplotlib.pyplot as plt
from os.path import join, basename, dirname, exists
import shutil
from utils_preprocess import *
import utils
import sys
import socket
from glob import glob
from itertools import islice, chain # a batch iterator
home = os.environ['HOME']

'''
python main.py -train -batchsize=50 -rnnsteps=32 -noartifact -beamsearch -name=dense_128_32_noartifact_beamsearch_prt 
# change the name to link to different model

'''

# basic operations
parser = argparse.ArgumentParser()
parser.add_argument("-name", default='debug', type=str, help="name of the log")
parser.add_argument("-gpu", default='0', type=str, help="gpu numbers")
parser.add_argument("-train", help="train the NN", action="store_true")
parser.add_argument("-validate", help="validate the NN", action="store_true")
parser.add_argument("-transfer", action="store_true")
parser.add_argument("-batchesTrained", default=0, type=int, help='number of batches already trained (for lr schedule)')
# beam search
parser.add_argument("-beamsearch", help="use beam search instead of best path decoding", action="store_true")
parser.add_argument("-wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
# training hyperparam
parser.add_argument("-batchsize", default=50, type=int, help='batch size')
parser.add_argument("-lrInit", default=1e-2, type=float, help='initial learning rate')
parser.add_argument("-optimizer", default='rmsprop', help="adam, rmsprop, momentum")
parser.add_argument("-wdec", default=1e-4, type=float, help='weight decay')
parser.add_argument("-lrDrop1", default=10, type=int, help='step to drop lr by 10 first time')
parser.add_argument("-lrDrop2", default=1000, type=int, help='step to drop lr by 10 sexond time')
parser.add_argument("-epochEnd", default=40, type=int, help='end after this many epochs')
# trainset hyperparam
parser.add_argument("-noncustom", help="noncustom (original) augmentation technique", action="store_true")
parser.add_argument("-noartifact", help="dont insert artifcats", action="store_true")
parser.add_argument("-iam", help='use iam dataset', action='store_true')
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
parser.add_argument("-crop_r1", default=3, type=int)
parser.add_argument("-crop_r2", default=28, type=int)
parser.add_argument("-crop_c1", default=10, type=int)
parser.add_argument("-crop_c2", default=115, type=int)
# filepaths
parser.add_argument("-dataroot", default='/root/datasets', type=str)
parser.add_argument("-ckptroot", default='/root/ckpt', type=str)
parser.add_argument("-urlTransferFrom", default=None, type=str)

# new added by yike
# inference
parser.add_argument("-bounding_box_batch_size", default=10, type=int,help='define how many line bounding box would be processed in one batch') # 


args = parser.parse_args()

name = args.name
ckptroot = join(home, 'ckpt')
args.ckptpath = join(ckptroot, name)

def batchnize(iterable, size):
  
  sourceiter = iter(iterable)
  while True:
    batchiter = islice(sourceiter, size)
    yield chain([batchiter.__next__()], batchiter)

def segmentation_wraper(img_path): # yike: should revise a little if the input is already cv2 images
  
  img = prepareImg(cv2.imread(img_path), 50)
  res = wordSegmentation(args,img, kernelSize=25, sigma=11, theta=7, minArea=200)
  boxes,images=zip(*res)
  
  return images


def main():
  #print(args)
  
  '''
  dumy loading data: detected line data, please revise this part according to real detected data
  this part might need rewriting upon real detected result data
  '''
  
  imgFiles=glob('/root/datasets/img_print_100000_en/**.jpg')
  boundingbox_lst=  [None for i in range(len(imgFiles))]# dumy data, boundingbox info, from detection result, would be used to aggregate recognized 
  
  detect_rlt=zip(imgFiles,boundingbox_lst)
  
  # dummy data construct complete org_imgs, bd_boxes
  
  '''
  Load Model, dummy model for now
  '''
  decoderType = DecoderType.BestPath
  if args.beamsearch:
    decoderType = DecoderType.BeamSearch
  elif args.wordbeamsearch:
    decoderType = DecoderType.WordBeamSearch

  model = Model(args, open(join(args.ckptpath, 'charList.txt')).read(), decoderType, mustRestore=True)
  '''
  Segmentation & Recognition, using dummy model
  '''

  
  ct=0
  rlt_all=[]
  for batch in batchnize(iterable=detect_rlt,size=args.bounding_box_batch_size):
    org_img_paths,bd_boxes=zip(*list(batch))
    ct=ct+1
    line_image_lists= list(map(segmentation_wraper,org_img_paths)) # assume segmentation works, risky
    line_cum_lens=list(np.cumsum(list(map(lambda l: len(l),line_image_lists))))
    bt_size=len(line_cum_lens)
    line_cum_lens.insert(0,0)
    #print(bt_size)
    #print(line_cum_lens)
    merged=list(chain.from_iterable(line_image_lists))
    recognized = model.inferBatch(merged)
    
    for idx in range(bt_size):
      rlt_text_line=' '.join(recognized[line_cum_lens[idx]:line_cum_lens[idx+1]])
      gt_text_line= str(basename(org_img_paths[idx])[:-4])
      rlt_all.append((rlt_text_line,gt_text_line,bd_boxes[idx]))
    #if ct==8:
      #pass
      #print(len(line_image_lists))
      #print(len(line_cum_lens))
      #print(line_cum_lens[0])
      #print(len(line_image_lists[0]))
      #print(line_image_lists[0][0].shape)
      #print(len(merged))
      #print(merged[0].shape)
      #print(len(recognized))
      #print(recognized[1])
      #print(org_img_paths)
      #print(recognized)
      #print(line_cum_lens)
      
    if ct>10:
      for tp in rlt_all:
        print(tp)
      break
  

  
  
  
  print('so far so good')
 

'''
	# read input images from 'in' directory
	imgFiles = os.listdir('../data/')
	for (i,f) in enumerate(imgFiles):
		print('Segmenting words of sample %s'%f)
		
		# read image, prepare it by resizing it to fixed height and converting it to grayscale
		img = prepareImg(cv2.imread('../data/%s'%f), 50)
		
		# execute segmentation with given parameters
		# -kernelSize: size of filter kernel (odd integer)
		# -sigma: standard deviation of Gaussian function used for filter kernel
		# -theta: approximated width/height ratio of words, filter function is distorted by this factor
		# - minArea: ignore word candidates smaller than specified area
		res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
		
		# write output to 'out/inputFileName' directory
		if not os.path.exists('../out/%s'%f):
			os.mkdir('../out/%s'%f)
		
		# iterate over all segmented words
		print('Segmented into %d words'%len(res))
		for (j, w) in enumerate(res):
			(wordBox, wordImg) = w
			(x, y, w, h) = wordBox
			cv2.imwrite('../out/%s/%d.png'%(f, j), wordImg) # save word
			cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
		
		# output summary image with bounding boxes around words
		cv2.imwrite('../out/%s/summary.png'%f, img)
'''

if __name__ == '__main__':
	main()