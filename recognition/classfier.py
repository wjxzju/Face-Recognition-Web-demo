#!---* coding: utf-8 --*--
#!/usr/bin/python
import cv2
import dlib
import numpy as np
import sys
import os
import skimage
import sklearn.metrics.pairwise as pw
import caffe
import json

MAIN_DIR = '/home/wjx/work/Face-Recognition-Web-demo/recognition'

PREDICTOR_PATH = MAIN_DIR + '/model/shape_predictor_68_face_landmarks.dat'
BASEFILE = MAIN_DIR + '/baseline/base.jpg'
BASE_LANDMARK = MAIN_DIR + '/baseline/BASE_LANDMARK.txt'

MODEL_FILE = MAIN_DIR + '/model/LightenedCNN_B.caffemodel'
DEPLOY_FILE = MAIN_DIR + '/proto/LightenedCNN_B_deploy_ugrade.prototxt'

JSON_FILE = MAIN_DIR + '/data.json'
DATASET_DIR =MAIN_DIR + '/dataset/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class NoFaceError(Exception):
    '''
        No face detect in a picture, when occur this situation, we need to handle 
    '''
    def __init__(self,str):
        self.str = str
    def __str__(self):
        return self.str

def getPoints(landmark):
    '''
        when alignment, we need some point to be baseline
        choose 37 43 30 48 54 as the baseline point
    '''
    Points = np.float32([[landmark[37][0],landmark[37][1]],[landmark[43][0],landmark[43][1]],[landmark[30][0],landmark[30][1]],[landmark[48][0],landmark[48][1]],[landmark[54][0],landmark[54][1]]])
    return Points

def getlandmark(im):

    image = im.copy()
    rects = detector(image,1)
    if len(rects) == 0:
        raise NoFaceError("No face detect")
    cv2.rectangle(image,(rects[0].left(),rects[0].top()),(rects[0].right(),rects[0].bottom()),(0,255,0),2)
    landmark = np.array([[p.x, p.y] for p in predictor(image, rects[0]).parts()])
    for j in xrange(landmark.shape[0]):
        pos = (landmark[j][0],landmark[j][1])
        cv2.putText(image, str(j), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(image, pos, 1, color=(255, 0, 0))

    return landmark,image


def compute_affine_transform(refpoints, points, w = None):
    '''
    计算仿射变换矩阵
    '''
    if w == None: #每个关键点的权重
        w = [1] * (len(points) * 2)
    assert(len(w) == 2*len(points))
    y = []
    for n, p in enumerate(refpoints):
        y += [p[0]/w[n*2], p[1]/w[n*2+1]]
    A = []
    for n, p in enumerate(points):
        A.extend([ [p[0]/w[n*2], p[1]/w[n*2], 0, 0, 1/w[n*2], 0], [0, 0, p[0]/w[n*2+1], p[1]/w[n*2+1], 0, 1/w[n*2+1]] ])
    
    lstsq = cv2.solve(np.array(A), np.array(y), flags=cv2.DECOMP_SVD)
    h11, h12, h21, h22, dx, dy = lstsq[1]
    #err = 0#lstsq[1]

    #R = np.array([[h11, h12, dx], [h21, h22, dy]])
    # The row above works too - but creates a redundant dimension
    R = np.array([[h11[0], h12[0], dx[0]], [h21[0], h22[0], dy[0]]])
    return R

def alignment(image):

    base_landmark = np.loadtxt(BASE_LANDMARK)
    base_landmark *= image.shape[0]
    
    try:
        landmark,image_show = getlandmark(image)
    except NoFaceError, e:
        raise e
    
    srcPoints = getPoints(landmark)
    dstPoints = getPoints(base_landmark)
    
    M =compute_affine_transform(dstPoints,srcPoints)
    image2 = cv2.warpAffine(image,M,(250,250))
    rects = detector(image2,2)
    if len(rects) == 0:
       raise NoFaceError("No face detect")

    return image2[rects[0].top():rects[0].bottom(),rects[0].left():rects[0].right()]

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


def checkjson(jsonfile):
    print "----------------------------------"
    print "        check json file           "
    print "----------------------------------"
    with open(jsonfile,'r') as f:
        data_dict = json.load(f)
        for dir in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR,dir)):
                if dir not in data_dict.keys():
                    return False
    return True

def updatejson(jsonfile,net):
    print "----------------------------------"
    print "       update json file           "
    print "----------------------------------"
    data_dict = {}
    with open(jsonfile,'r') as f:
        data_dict = json.load(f)
        for dir in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR,dir)):
                if dir not in data_dict.keys():
                    subdir = os.path.join(DATASET_DIR,dir)
                    data_list  =[] 
                    for file in os.listdir(subdir):
                        if  "_face.jpg" in file:
                            image = os.path.join(subdir,file)
                            data_list.append(compute_feature(image,net).tolist())
                    data_dict[dir] = data_list
                    data_array = np.asarray(data_list,)

    with open(jsonfile,'w') as f:
        json.dump(data_dict,f)


class Classfier(object):
    
    def __init__(self):
        self.net = caffe.Net(DEPLOY_FILE,MODEL_FILE,caffe.TEST)
        self.threshold = 0.35

    def checkdataset(self):
        for dir in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR,dir)):
                subdir = os.path.join(DATASET_DIR,dir)
                for file in os.listdir(subdir):
                    if not "_face.jpg" in file:
                        facefilename = file.split('.')[0]+'_face.jpg'
                        if not os.path.isfile(os.path.join(subdir,facefilename)):
                            facefilename = self.alignment(os.path.join(subdir,file))
    def check(self):
        self.checkdataset()
        if not os.path.exists(JSON_FILE):
            with open(JSON_FILE,'w') as f:
                data_dict = {}
                json.dump(data_dict,f)
            updatejson(JSON_FILE,self.net)

        elif not checkjson(JSON_FILE):
            updatejson(JSON_FILE,self.net)
        print "----------------------------------"
        print "       classfier check done       "
        print "----------------------------------"

    def getpersonlist(self):
        personlist = []
        for dir in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR,dir)):
                personlist.append(dir)

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

    def alignment(self,imagefile):
        facefilename = imagefile.split('.')[0]+'_face.jpg'
        image = cv2.imread(imagefile)
        image = cv2.resize(image,(250,250))
        try:
             face = alignment(image)
        except NoFaceError, e:
            return None
       
        face = cv2.resize(face,(128,128))
        cv2.imwrite(facefilename,face)
        return facefilename

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


