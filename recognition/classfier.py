#!---* coding: utf-8 --*--
#!/usr/bin/python
import numpy as np
import sys
import os
import skimage
import sklearn.metrics.pairwise as pw
import caffe
import json
from dataset import db


MAIN_DIR = '/home/wjx/work/Face-Recognition-Web-demo/recognition'

MODEL_FILE = MAIN_DIR + '/model/LightenedCNN_B.caffemodel'
DEPLOY_FILE = MAIN_DIR + '/proto/LightenedCNN_B_deploy_ugrade.prototxt'

JSON_FILE = MAIN_DIR + '/data.json'
DATASET_DIR =MAIN_DIR + '/dataset/'

DATABASE = db()


def read_image(filelist):

    X=np.empty((1,1,128,128))
    word=filelist.split('\n')
    filename=word[0]
    im1=skimage.io.imread(filename,as_grey=True)
  
    #归一化
    image =skimage.transform.resize(im1,(128,128))*255
    X[0,:,:,:]=image[:,:]
   
    X = X* 0.00390625
    return X

def compute_feature(path,net):
    X=read_image(path)

    out = net.forward_all(data = X)

    feature = np.float64(out['fc1'])
    return feature


def updatejson(jsonfile,net):
    print "----------------------------------"
    print "       update json file           "
    print "----------------------------------"
    data_dict = {}
    with open(jsonfile,'r') as f:
        data_dict = json.load(f)
        for dir in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR,dir)):
                subdir = os.path.join(DATASET_DIR,dir)
                data_list  =[] 
                #print dir
                if dir not in data_dict.keys():
                    for file in os.listdir(subdir):
                        if  "_face.jpg" in file:
                            image = os.path.join(subdir,file)
                            data_list.append(compute_feature(image,net).tolist())
                    data_dict[dir] = data_list
                else:
                    data_list = data_dict[dir]
                    #print DATABASE.getnum(dir)
                    if len(data_list) < DATABASE.getnum(dir):
                        print "num is not equal, update"
                        file = str(DATABASE.getnum(dir)-1)+'_face.jpg'
                        image = os.path.join(subdir,file)
                        if os.path.isfile(image):
                            data_list.append(compute_feature(image,net).tolist())  
                    data_dict[dir] = data_list
    with open(jsonfile,'w') as f:
        json.dump(data_dict,f)


class Classfier(object):
    
    def __init__(self):
        self.net = caffe.Net(DEPLOY_FILE,MODEL_FILE,caffe.TEST)
        self.threshold = 0.33

    def check(self):
        DATABASE.update_dataset()
        if not os.path.exists(JSON_FILE):
            with open(JSON_FILE,'w') as f:
                data_dict = {}
                json.dump(data_dict,f)
            updatejson(JSON_FILE,self.net)

        else:
            updatejson(JSON_FILE,self.net)
        print "----------------------------------"
        print "       classfier check done       "
        print "----------------------------------"


    def verification(self,facefilename):
        feature = compute_feature(facefilename,self.net)
        with open(JSON_FILE,'r') as f:
            data_dict = json.load(f)
            verdict = {}  # a dict use to store the verification result 
            for key in data_dict:
                data_list = data_dict[key]
                data_array = np.asarray(data_list,dtype=np.float32)
                compare_result = np.empty((len(data_list)))
                for i in xrange(len(data_list)):
                    distance  = pw.pairwise_distances(feature, data_array[i,:,:], metric="cosine")[0][0]
                    compare_result[i] = distance
                verdict[key] = compare_result.mean()
        return verdict

    def recognition(self,facefilename):
        print "----------------------------------"
        print "       start recognition          "
        print "----------------------------------"
        result = self.verification(facefilename)
        result = sorted(result.items(), key=lambda d:d[1])
        personid = ''
        if result[0][1] < self.threshold:
            personid = result[0][0]

        print "----------------------------------"
        print "   person name         "+ personid
        print "----------------------------------"
        return personid

if __name__ == '__main__':
    classfier = Classfier()
    classfier.check()
