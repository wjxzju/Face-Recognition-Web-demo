#!---* coding: utf-8 --*--
#!/usr/bin/python
import cv2
import dlib
import numpy as np
import sys
import os
from skimage import transform as tf
import skimage
import sklearn.metrics.pairwise as pw
import caffe
import json

PREDICTOR_PATH = './model/shape_predictor_68_face_landmarks.dat'
BASEFILE = './baseline/base.jpg'
BASE_LANDMARK = './baseline/BASE_LANDMARK.txt'

MODEL_FILE = './model/LightenedCNN_B.caffemodel'
DEPLOY_FILE = './proto/LightenedCNN_B_deploy_ugrade.prototxt'
DATASET_DIR = './dataset/'
JSON_FILE = './data.json'

net=caffe.Net(DEPLOY_FILE,MODEL_FILE,caffe.TEST)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def getPoints(landmark):
    '''
    choose 37 43 30 48 54 as the baseline point
    '''
    
    Points = np.float32([[landmark[37][0],landmark[37][1]],[landmark[43][0],landmark[43][1]],[landmark[30][0],landmark[30][1]],[landmark[48][0],landmark[48][1]],[landmark[54][0],landmark[54][1]]])
    return Points


def getlandmark(im):
    image = im.copy()
   
    rects = detector(image,1)
    if len(rects) == 0:
        print "No face detect in origin picture!"
        sys.exit(0)
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
    if w == None:#每个关键点的权重
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
    # draw the face
    landmark,image_show = getlandmark(image)
    srcPoints = getPoints(landmark)
    dstPoints = getPoints(base_landmark)
    
    M =compute_affine_transform(dstPoints,srcPoints)
    image2 = cv2.warpAffine(image,M,(250,250))
    rects = detector(image2,2)
    if len(rects) == 0:
        print "After affine, no face detect"
        sys.exit(0)

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

def compar_pic(path1,path2):
    global net
    #加载验证图片
    X=read_image(path1)
    
    out = net.forward_all(data = X)
   
    feature1 = np.float64(out['fc1'])
  
    X=read_image(path2)
    
    out = net.forward_all(data = X)
   
    feature2 = np.float64(out['fc1'])
    
    #求两个特征向量的cos值,并作为是否相似的依据
    predicts = pw.pairwise_distances(feature1, feature2, metric="cosine")
    return  predicts[0][0]

def compute_feature(path):
    global net
    X=read_image(path)
    out = net.forward_all(data = X)
    feature = np.float64(out['fc1'])

    return feature.tolist()

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print "Usage:"
    #     print "python main.py picture1, picture2"
    #     sys.exit(0)

    # picture1 = sys.argv[1]
    # print 'picture1 -----> ',picture1
    # face1 = picture1[:-4]+'_face.jpg'
    # if not os.path.isfile(face1):
    #     image = cv2.imread(picture1)
    #     image = cv2.resize(image,(250,250))
    #     face = alignment(image)
    #     face = cv2.resize(face,(128,128))
    #     cv2.imwrite(face1,face)

    # picture2 = sys.argv[2]
    # print 'picture2 -----> ',picture2
    # face2 = picture2[:-4]+'_face.jpg'
    # if not os.path.isfile(face2):
    #     image = cv2.imread(picture2)
    #     image = cv2.resize(image,(250,250))
    #     face = alignment(image)
    #     face = cv2.resize(face,(128,128))
    #     cv2.imwrite(face2,face)

    # result=compar_pic(face1,face2)
    # print "%s和%s两张图片的cosine distance:%f\n\n"%(face1,face2,result)
    # if result<=0.3:
    #     print '是一个人!!!!\n\n'
    # else:
    #     print '不是同一个人!!!!\n\n'
    
    # data_dict= {}
    # for dir in os.listdir(DATASET_DIR):
    #     subdir = os.path.join(DATASET_DIR,dir)
    #     data_list  =[] 
    #     if dir  == 'wjx':
    #         for file in os.listdir(subdir):
    #                 if  "_face.jpg" in file:
    #                     image = os.path.join(subdir,file)
    #                     data_list.append(compute_feature(image))
    #                     data_dict[dir] = data_list
    # with open(JSON_FILE,'w') as f:
    #     json.dump(data_dict,f)

    with open(JSON_FILE,'r') as f:
        data_dict = json.load(f)
        for dir in os.listdir(DATASET_DIR):
            datalist = data_dict[dir]
            # for i in xrange(len(datalist):
            array = np.asarray(datalist,dtype=np.float32)
            print array.shape




