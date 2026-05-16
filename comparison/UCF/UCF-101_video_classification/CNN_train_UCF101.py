"""
Train on images from UCFdata.py split.
- train: FF++ only
- test/validation: external datasets
Use keras 2+ and tensorflow 1+
"""
import math
import os
import pandas as pd

from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from UCFdata import DataSet

data = DataSet()

CLASSES = data.classes
BATCH_SIZE = 32
TARGET_SIZE = (299, 299)

# Helper: Save the min val_loss model in each epoch.
checkpointer = ModelCheckpoint(
    filepath='./data/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True
)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir='./data/logs/')


def samples_to_frame_dataframe(samples, split_name):
    """
    UCFdata.py의 sample = [split, cls, video_name, n_frames, vid_dir]
    구조를 프레임 단위 dataframe으로 펼친다.
    """
    rows = []

    for sample in samples:
        label = sample[1]
        frame_list = data.get_frames_for_sample(sample)

        for frame_path in frame_list:
            rows.append({
                "filename": frame_path,
                "class": label,
                "split": split_name
            })

    df = pd.DataFrame(rows)
    return df


def get_generators():
    train_samples, test_samples = data.split_train_test()

    train_df = samples_to_frame_dataframe(train_samples, "train")
    test_df = samples_to_frame_dataframe(test_samples, "test")

    print("Train videos:", len(train_samples))
    print("Test videos :", len(test_samples))
    print("Train frames:", len(train_df))
    print("Test frames :", len(test_df))

    if len(train_df) == 0:
        raise ValueError("Train dataframe is empty. Check FF++ path in UCFdata.py")
    if len(test_df) == 0:
        raise ValueError("Test dataframe is empty. Check external dataset paths in UCFdata.py")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator


def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # binary(real/fake) classifier
    predictions = Dense(len(CLASSES), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def fine_tune_inception_layer(model):
    """After we fine-tune the dense layers, train deeper."""
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model


def train_model(model, nb_epoch, generators, callbacks=None):
    if callbacks is None:
        callbacks = []

    train_generator, validation_generator = generators

    steps_per_epoch = max(1, math.ceil(train_generator.samples / float(train_generator.batch_size)))
    validation_steps = max(1, math.ceil(validation_generator.samples / float(validation_generator.batch_size)))

    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=nb_epoch,
        callbacks=callbacks
    )
    return model


def main(weights_file=None):
    os.makedirs('./data/checkpoints', exist_ok=True)
    os.makedirs('./data/logs', exist_ok=True)

    model = get_model()
    generators = get_generators()

    if weights_file is None:
        print("Training top layers.")
        model = train_model(model, 10, generators)
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    model = fine_tune_inception_layer(model)
    model = train_model(
        model,
        1000,
        generators,
        [checkpointer, early_stopper, tensorboard]
    )


if __name__ == '__main__':
    weights_file = None
    main(weights_file)