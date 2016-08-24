#!/usr/bin/python 
#--*--coding: utf-8--*--
from flask_wtf import Form
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, StringField
from wtforms.validators import Required

class PhotoForm(Form):
    photo = FileField('', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png'], 'Images only!')])
    submit = SubmitField('Submit')

class RegForm(Form):
    name = StringField('Input your name',validators=[Required()])
    submit = SubmitField('submit')
        
        

