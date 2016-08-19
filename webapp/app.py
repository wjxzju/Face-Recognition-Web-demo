#!/usr/bin/python
#--*-coding: utf-8 --*-
import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, session, flash 
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import tornado.wsgi
import tornado.httpserver
import optparse

import sys

sys.path.append('../')

from recognition.classfier import *


UPLOAD_FOLDER = "/tmp/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'hard to guess string'

bootstrap = Bootstrap()
bootstrap.init_app(app)

classfier = Classfier()
classfier.check()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            facefilename = classfier.alignment(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if facefilename ==  None:
                flash('This picture can not find faces, please change')
            else:
                result = classfier.recognition(facefilename)
                print result
    return render_template('index.html')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)

    opts, args = parser.parse_args()
    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)