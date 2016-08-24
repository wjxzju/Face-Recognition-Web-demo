#!---* coding: utf-8 --*--
#!/usr/bin/python
import cv2
import dlib
import numpy as np
import os

MAIN_DIR = '/home/wjx/work/Face-Recognition-Web-demo/recognition'

PREDICTOR_PATH = MAIN_DIR + '/model/shape_predictor_68_face_landmarks.dat'
BASEFILE = MAIN_DIR + '/baseline/base.jpg'
BASE_LANDMARK = MAIN_DIR + '/baseline/BASE_LANDMARK.txt'

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
    rects = detector(image,2)
    if len(rects) == 0:
        raise NoFaceError("No face detect")
    # cv2.rectangle(image,(rects[0].left(),rects[0].top()),(rects[0].right(),rects[0].bottom()),(0,255,0),2)
    landmark = np.array([[p.x, p.y] for p in predictor(image, rects[0]).parts()])
    # for j in xrange(landmark.shape[0]):
    #     pos = (landmark[j][0],landmark[j][1])
        # cv2.putText(image, str(j), pos,
        #             fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        #             fontScale=0.4,
        #             color=(0, 0, 255))
        # cv2.circle(image, pos, 1, color=(255, 0, 0))

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


def alignment(imagefile):
    facefilename = imagefile.split('.')[0]+'_face.jpg'
    image = cv2.imread(imagefile)
    image = cv2.resize(image,(250,250))

    base_landmark = np.loadtxt(BASE_LANDMARK)
    base_landmark *= image.shape[0]
    
    try:
        landmark,image_show = getlandmark(image)
    except NoFaceError, e:
        return None
    
    srcPoints = getPoints(landmark)
    dstPoints = getPoints(base_landmark)
    
    M =compute_affine_transform(dstPoints,srcPoints)
    image2 = cv2.warpAffine(image,M,(250,250))
    rects = detector(image2,2)
    if len(rects) == 0:
       return None

    face =image2[rects[0].top():rects[0].bottom(),rects[0].left():rects[0].right()]
    face = cv2.resize(face,(128,128))
    cv2.imwrite(facefilename,face)
    return facefilename
