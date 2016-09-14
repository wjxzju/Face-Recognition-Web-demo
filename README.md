# Face-Recoginition-Web-demo
* Description: A web demo of face recognition.
* Mainly use `caffe` `flask` `dlib` `opencv` framworks.
* You must run this program in a linux server, and view this face recognition system by a browser on client, I suggest you use chrome.
* The System support 3 mainly functions, `register`, `upload picture recognition` and `real time face recognition`.
* When you first use this system, you should start on register, because there is no picture store on the database.
* The registered pictures stored in `recognition/dataset/*person_name*/**`

## How to start?
* Download `base.jpg`, `LightenedCNN_B.caffemodel`, `shape_predictor_68_face_landmarks.dat` from   
`链接：http://pan.baidu.com/s/1c1W8rJE 密码：223q`
* `cd webapp`, `python app.py` then the system start on server.
* View the system by the url `http://serverip:5000(default)` 


