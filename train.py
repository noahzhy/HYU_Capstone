import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import *
from keras.optimizers import *

from models.centernet import CenterNet
from models.data_gen import Generator


def get_random_lines(file_path, random_seed=7):
    with open(file_path) as f:
        lines = f.readlines()
    np.random.seed(random_seed)
    np.random.shuffle(lines)
    np.random.seed(None)
    return lines


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    return [c.strip() for c in class_names]


if __name__ == "__main__":
    input_shape = [512, 512, 3]

    class_path = 'datasets/voc_classes.txt'
    class_name = get_classes(class_path)
    num_class = len(class_name)
    annotation_path = '2007_train.txt'
    data_num = get_random_lines(annotation_path)
    val_split = 0.2
    num_val = int(len(data_num)*(1-val_split))
    num_train = len(data_num) * num_val

    backbone = "resnet50"

    learning_rate = 1e-3
    Batch_size = 128
    freeze_epoch = 50
    epoch = 100

    model_path = "datasets/centernet_resnet50_voc.h5"

    gen = Generator(batch_size, data_num[:num_train],
                    data_num[num_train:], input_shape, num_class)

    model = CenterNet(
        input_shape,
        num_classes=num_class,
        backbone=backbone,
        mode='train'
    )
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    checkpoint = ModelCheckpoint(
        'logs/ep{epoch:03d}_loss{loss:.3f}_val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=False,
        period=1
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    model.compile(
        loss={'centernet_loss': lambda y_true, y_pred: y_pred},
        optimizer=Adam(learning_rate)
    )

    model.fit_generator(
        gen.generate(True),
        steps_per_epoch=num_train//batch_size,
        validation_data=gen.generate(False),
        validation_steps=num_val//batch_size,
        epochs=epoch,
        verbose=1,
        initial_epoch=freeze_epoch,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping]
    )
