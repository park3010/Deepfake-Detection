# UCF_train.py
"""
Train on FF++ mtcnn frames directly.
- train/val: split from FF++ frames
- binary classification: real vs fake
"""

import os

GPU_ID = "1"   # 쓰고 싶은 GPU 번호
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import math
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# ───────── FF++ 디렉터리 맵 ──────────────────────────────────────
DATASETS = {
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures',
}

FFPP_ROOT = "/home/oem/deepfake/hdd"   # 네 환경에 맞게 수정
COMPRESSION = "raw"

CLASSES = ["real", "fake"]
BATCH_SIZE = 16
TARGET_SIZE = (299, 299)
RANDOM_STATE = 42

# Helper: Save the min val_loss model in each epoch.
checkpointer = ModelCheckpoint(
    filepath='/home/oem/deepfake/Ourmethod/comparison/_ckpt/ucf/train/inception.{epoch:03d}-{val_loss:.4f}.hdf5',
    verbose=1,
    save_best_only=True
)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=5, restore_best_weights=True)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir='/home/oem/deepfake/Ourmethod/comparison/_ckpt/ucf/log')


def collect_ffpp_frames(root_dir, compression='raw', selected_keys=None):
    """
    FF++ mtcnn 프레임을 직접 수집해서 dataframe 생성
    columns:
        - filename
        - class   : "real" / "fake"
        - label   : 0 / 1
        - source  : dataset key
    """
    rows = []

    if selected_keys is None:
        selected_keys = list(DATASETS.keys())

    for key in selected_keys:
        base = os.path.join(root_dir, DATASETS[key], compression, 'mtcnn')
        if not os.path.isdir(base):
            print(f"[WARN] Missing directory: {base}")
            continue

        label = 0 if 'original' in key else 1
        class_name = "real" if label == 0 else "fake"

        for sub, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rows.append({
                        "filename": os.path.join(sub, f),
                        "class": class_name,
                        "label": label,
                        "source": key,
                    })

    df = pd.DataFrame(rows)
    return df


def get_generators():
    df = collect_ffpp_frames(FFPP_ROOT, compression=COMPRESSION)

    print("Total frames:", len(df))
    if len(df) == 0:
        raise ValueError("No FF++ frames found. Check FFPP_ROOT / COMPRESSION.")

    # 프레임 단위 분할
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["class"]
    )

    print("Train frames:", len(train_df))
    print("Val frames  :", len(val_df))
    print("Train class counts:\n", train_df["class"].value_counts())
    print("Val class counts:\n", val_df["class"].value_counts())

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)

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

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
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
    base_model = InceptionV3(weights=weights, include_top=False, input_shape=(299, 299, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(CLASSES), activation='softmax')(x)

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
    """After training top layers, unfreeze deeper layers."""
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    model.compile(
        optimizer=SGD(learning_rate=0.0001, momentum=0.9),
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

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=nb_epoch,
        callbacks=callbacks
    )
    return model


def main(weights_file=None):
    os.makedirs('/home/oem/deepfake/Ourmethod/comparison/_ckpt/ucf/train', exist_ok=True)
    os.makedirs('/home/oem/deepfake/Ourmethod/comparison/_ckpt/ucf/log', exist_ok=True)

    model = get_model()
    generators = get_generators()

    if weights_file is None:
        print("Training top layers.")
        model = train_model(model, 5, generators)
    else:
        print(f"Loading saved model: {weights_file}")
        model.load_weights(weights_file)

    model = fine_tune_inception_layer(model)
    model = train_model(
        model,
        2000,
        generators,
        [checkpointer, early_stopper, tensorboard]
    )


if __name__ == '__main__':
    main()