import contextlib
import os
import shutil
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 12})

PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, "Dataset/Mushrooms/")


@contextlib.contextmanager
def change_dir(path):
    last_dir = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(last_dir)


def create_directories(dir_list):
    for folder in dir_list:
        Path(folder).mkdir(parents=True, exist_ok=True)


def prepare_image_data():
    # Temporary folders for training, validation and test images:
    create_directories(['kaggle', 'kaggle/temp'])
    with change_dir('kaggle/temp'):
        create_directories(['train', 'valid', 'test'])

        for sub_dir in os.listdir(DATA_PATH):
            # Making a list of all files in current sub_dir:
            original_path = os.path.join(DATA_PATH, sub_dir)
            original_data = os.listdir(original_path)

            # Number of samples in each group:
            n_samples = len(original_data)
            train_samples = int(n_samples * 0.75)
            valid_samples = int(n_samples * 0.9)

            folder_data = {
                "train": [train_samples],
                "valid": [train_samples, valid_samples],
                "test": [valid_samples, n_samples]
            }

            for folder, data in folder_data.items():
                # New class sub_dir for training:
                with change_dir(folder):
                    create_directories([sub_dir])
                    # Training images:
                    for image_file in range(*data):
                        original_file = os.path.join(original_path, original_data[image_file])
                        new_file = os.path.join(PROJECT_PATH,
                                                'kaggle/temp',
                                                folder,
                                                sub_dir,
                                                original_data[image_file])
                        shutil.copyfile(original_file, new_file)


def train_neural_network():
    # "Prepare data for ML model:"
    train_generator = ImageDataGenerator(preprocessing_function=preprocess_input) \
        .flow_from_directory(directory='d:/University_projects/AWD/kaggle/temp/train',
                             target_size=(224, 224),
                             class_mode='categorical',
                             batch_size=100)
    valid_generator = ImageDataGenerator(preprocessing_function=preprocess_input) \
        .flow_from_directory(directory='d:/University_projects/AWD/kaggle/temp/valid',
                             target_size=(224, 224),
                             class_mode='categorical',
                             batch_size=100)

    # Load ResNet50 model:
    print("Loading ResNet model...")
    resnet_model = ResNet50(weights="imagenet")
    resnet_model.summary()

    # Extract all layers up to the final average pooling layer:
    print("Extracting all layers...")
    last_layer = resnet_model.get_layer("avg_pool")
    resnet_layers = Model(inputs=resnet_model.inputs, outputs=last_layer.output)

    # Construct a model with the final dense layer for 9 classes:
    print("Constructing a model...")
    model = Sequential()
    model.add(resnet_layers)
    model.add(Dense(9, activation="softmax"))

    # Make all the layers from the original ResNet model untrainable:
    model.layers[0].trainable = False

    # Metrics and optimizer:
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Check the structure of new model:
    model.summary()

    # Introduce callbacks to be exercised during training:
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                                  mode='max', cooldown=2, patience=2, min_lr=0)

    # Train new model:"
    print("Training model")
    history = model.fit(train_generator, validation_data=valid_generator,
                        epochs=50, verbose=2, callbacks=[reduce_lr, early_stop])

    # saving model - do apki
    print("Saving model...")
    model.save('d:/University_projects/AWD/resnet50-mushrooms.model')

    return history


def show_model_parameters(history_model):
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input) \
        .flow_from_directory(directory='d:/University_projects/AWD/kaggle/temp/test',
                             target_size=(224, 224),
                             class_mode='categorical',
                             batch_size=100)

    loaded_model = tf.keras.models.load_model('resnet50-mushrooms.model')
    loss, accuracy = loaded_model.evaluate(test_generator, verbose=2)
    print(f'Model performance on test images:\nAccuracy = {accuracy}\nLoss = {loss}')

    # Loss during training:
    history_frame = pd.DataFrame(history_model.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()

    # Accuracy during training:
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.show()


def identify_picture(img_path):
    loaded_model = tf.keras.models.load_model('resnet50-mushrooms.model')

    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.show()

    x_array = image.img_to_array(img)
    x_array = np.expand_dims(x_array, axis=0)
    images = np.vstack([x_array])
    val = loaded_model.predict(images)
    print(val)


if __name__ == '__main__':
    # prepare_image_data()
    model_history = train_neural_network()
    show_model_parameters(model_history)
