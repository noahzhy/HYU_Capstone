import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import *
from keras.optimizers import *

from models.snet_centernet import *
from models.centernet_training import *
from tensorflow.keras.utils import plot_model
# from models.data_gen import Generator


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


if __name__ == "__main__":
    classes_path = 'datasets/coco_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    annotation_path = 'train.txt'
    val_split = 0.1

    with open(annotation_path) as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    logging = TensorBoard(log_dir="logs/")
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=10)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history = LossHistory("logs/")

    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    INIT_EPOCH = 0
    EPOCH = 50

    input_shape = (512, 512, 3)
    gen = Generator(
        BATCH_SIZE,
        lines[:num_train],
        lines[num_train:],
        input_shape,
        num_classes
    )

    epoch_size = num_train // BATCH_SIZE
    epoch_size_val = num_val // BATCH_SIZE

    model = snet_x(input_shape, num_class=num_classes, scale=49)
    plot_model(model, to_file='snet_centernet.png', show_layer_names=True, show_shapes=True)
    losses = [focal_loss, reg_l1_loss, reg_l1_loss]
    model.compile(optimizer=Adam(LEARNING_RATE), loss=losses)
    model.summary()

    # model.compile(
    #     loss={'centernet_loss': lambda y_true, y_pred: y_pred},
    #     optimizer=keras.optimizers.Adam(Lr)
    # )

    model.fit(
        gen.generate(True),
        steps_per_epoch=epoch_size,
        validation_data=gen.generate(False),
        validation_steps=epoch_size_val,
        epochs=EPOCH,
        verbose=1,
        initial_epoch=INIT_EPOCH,
        callbacks=[logging, checkpoint, reduce_lr,
                   early_stopping, loss_history]
    )
