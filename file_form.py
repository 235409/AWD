from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField


class FileForm(FlaskForm):
    picture = FileField('Dodaj zdjęcie', validators=[FileAllowed(['jpg', 'png'])])
    submit = SubmitField("Identyfikuj")
