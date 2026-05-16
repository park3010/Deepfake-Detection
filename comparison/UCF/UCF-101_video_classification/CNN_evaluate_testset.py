"""
Classify/evaluate test images set through our CNN.
Use keras 2+ and tensorflow 1+
- train: FF++ only
- test : external datasets from UCFdata.py
"""
import math
import pandas as pd

from UCFdata import DataSet
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

data = DataSet()
CLASSES = data.classes
BATCH_SIZE = 32
TARGET_SIZE = (299, 299)


def samples_to_frame_dataframe(samples, split_name):
    """
    UCFdata.py sample format:
    [split, class_name, video_name, n_frames, vid_dir]
    -> frame-level dataframe
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

    return pd.DataFrame(rows)


def build_test_generator():
    _, test_samples = data.split_train_test()

    test_df = samples_to_frame_dataframe(test_samples, "test")

    print("Test videos :", len(test_samples))
    print("Test frames :", len(test_df))

    if len(test_df) == 0:
        raise ValueError("Test dataframe is empty. Check dataset paths in UCFdata.py")

    test_data_gen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_data_gen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical',
        shuffle=False
    )

    return test_generator


def main(weights_file='data/checkpoints/inception.057-1.16.hdf5'):
    test_generator = build_test_generator()

    model = load_model(weights_file)

    steps = max(1, math.ceil(test_generator.samples / float(test_generator.batch_size)))

    results = model.evaluate_generator(
        generator=test_generator,
        steps=steps
    )

    print("Evaluation results:", results)
    print("Metrics:", model.metrics_names)


if __name__ == '__main__':
    main()