import glob
import os
import random

from flask import Flask, render_template, flash, session, redirect, url_for
from file_form import FileForm
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['SECRET_KEY'] = '13643ac98c6a09fec17a9b5de33b0408'


MUSHROOMS_DICT = {
    0: "Agaricus",
    1: "Amanita",
    2: "Boletus",
    3: "Cortinarius",
    4: "Entoloma",
    5: "Hygrocybe",
    6: "Lactarius",
    7: "Russula",
    8: "Suillus"
}


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


def get_mushroom(file_path):
    model = tf.keras.models.load_model('resnet50-mushrooms.model')

    tf_img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(tf_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.vstack([img_array])
    output = model.predict(img_array)

    array = output[0]
    index = array.argmax(axis=0)
    return MUSHROOMS_DICT[index], round(array[index] * 100, 2)


def save_file(form_picture):
    for rfile in glob.glob('static/inserted_img/*'):
        os.remove(rfile)
    _, extension = os.path.splitext(form_picture.filename)
    file_path = os.path.join("static/inserted_img", f"inserted_file_{random.randint(100, 1000)}{extension}")
    img = Image.open(form_picture)
    img.save(file_path)
    session['file_path'] = file_path


@app.route('/', methods=['POST', 'GET'])
def insert_page():
    form = FileForm()
    if form.validate_on_submit():
        if form.picture.data:
            save_file(form.picture.data)
            return redirect(url_for('result_page'))
        else:
            flash("Musisz podać plik aby identyfikować!", "danger")
    return render_template('insert_page.jinja2', form=form)


@app.route('/result')
def result_page():
    mushroom_img = str(session['file_path']).replace("\\", "/")
    mushroom_name, value = get_mushroom(mushroom_img)
    return render_template('result_page.jinja2',
                           mushroom_picture=mushroom_img.lstrip('static'),
                           predicted_picture=f"images/{mushroom_name.lower()}.jpg",
                           percent=str(value),
                           mushroom_name=mushroom_name)


@app.route('/authors')
def authors_page():
    return render_template('authors_page.jinja2')


if __name__ == '__main__':
    app.run()
