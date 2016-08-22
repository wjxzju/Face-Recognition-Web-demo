#!/usr/bin/python 
#--*--coding: utf-8--*--
from flask_wtf import Form
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField

class PhotoForm(Form):
    photo = FileField('Image', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png'], 'Images only!')])
    submit = SubmitField('Submit')

    

        

