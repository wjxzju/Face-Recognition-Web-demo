#!---* coding: utf-8 --*--
#!/usr/bin/python
import os
import shutil
from image import alignment

MAIN_DIR = '/home/wjx/work/Face-Recognition-Web-demo/recognition'
DATASET_DIR =MAIN_DIR + '/dataset/'

class db(object):
    """docstring for db"""
    def __init__(self):
        pass

    def getpersonlist(self):
        personlist = []
        for dir in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR,dir)):
                personlist.append(dir)
        return personlist

    def getnum(self,name):
        num = 0
        for file in os.listdir(os.path.join(DATASET_DIR,name)):
            if '_face.jpg' in file:
                num += 1
        return num

    def addperson(self,file,name):
        personlist = self.getpersonlist()
        if name not in personlist:
            newfolder= os.path.join(DATASET_DIR,name)
            os.mkdir(newfolder)
            if os.path.isdir(newfolder):
                dstfile = os.path.join(newfolder,'0_face.jpg')
                shutil.copyfile(file,dstfile)
        else:
            num = self.getnum(name)
            if num >=10:
                print "----------------------------------"
                print "       the database "+ name +" is full    "
                print "----------------------------------"
                return None
            dstfile = os.path.join(os.path.join(DATASET_DIR,name),str(num)+'_face.jpg')
            shutil.copyfile(file,dstfile)
        print "----------------------------------"
        print "       add person done            "
        print "----------------------------------"

    def update_dataset(self):
        for dir in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR,dir)):
                subdir = os.path.join(DATASET_DIR,dir)
                for file in os.listdir(subdir):
                    if not "_face.jpg" in file:
                        facefilename = file.split('.')[0]+'_face.jpg'
                        if not os.path.isfile(os.path.join(subdir,facefilename)):
                            facefilename = alignment(os.path.join(subdir,file))
if __name__ == '__main__':
    db = db()
    file = '/tmp/realtime_face.jpg'
    db.addperson(file,'he')

