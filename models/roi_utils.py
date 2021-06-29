from keras.layers import *
from keras import backend as K

import tensorflow as tf
import numpy as np


class PSRoiAlignPooling(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    """

    def __init__(self, pool_size, num_rois, alpha, **kwargs):
        self.dim_ordering = 'tf'
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.alpha_channels = alpha

        super(PSRoiAlignPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.alpha_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)
        total_bins = 1
        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        # because crop_size of tf.crop_and_resize requires 1-D tensor, we use uniform length
        bin_crop_size = []
        for num_bins, crop_dim in zip((7, 7), (14, 14)):
            assert num_bins >= 1
            assert crop_dim % num_bins == 0
            total_bins *= num_bins
            bin_crop_size.append(crop_dim // num_bins)

        xmin, ymin, xmax, ymax = tf.unstack(rois[0], axis=1)
        spatial_bins_y =  spatial_bins_x = 7
        step_y = (ymax - ymin) / spatial_bins_y
        step_x = (xmax - xmin) / spatial_bins_x

        # gen bins
        position_sensitive_boxes = []
        for bin_x in range(self.pool_size): 
            for bin_y in range(self.pool_size):
                box_coordinates = [
                    ymin + bin_y * step_y,
                    xmin + bin_x * step_x,
                    ymin + (bin_y + 1) * step_y,
                    xmin + (bin_x + 1) * step_x 
                ]
                position_sensitive_boxes.append(tf.stack(box_coordinates, axis=1))
        
        img_splits = tf.split(img, num_or_size_splits=total_bins, axis=3)
        box_image_indices = np.zeros(self.num_rois)

        feature_crops = []
        for split, box in zip(img_splits, position_sensitive_boxes):
            #assert box.shape[0] == box_image_indices.shape[0], "Psroi box number doesn't match roi box indices!"
            crop = tf.image.crop_and_resize(
                split, box, box_image_indices,
                bin_crop_size, method='bilinear'
            )
            # shape [num_boxes, crop_height/spatial_bins_y, crop_width/spatial_bins_x, depth/total_bins]

            # do max pooling over spatial positions within the bin
            crop = tf.reduce_max(crop, axis=[1, 2])
            crop = tf.expand_dims(crop, 1)
            # shape [num_boxes, 1, depth/total_bins]

            feature_crops.append(crop)

        final_output = K.concatenate(feature_crops, axis=1)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 5)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.alpha_channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RoiPoolingConv(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    """

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = 'tf'
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

