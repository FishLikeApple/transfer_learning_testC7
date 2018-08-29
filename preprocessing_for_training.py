"""
此文件中的代码用于对原始图像进行预处理，基本和书上一致。
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import inference

def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0., 1.)

def preprocessing_for_training(image, unified_image_shape, bbox):
    # 查看是否存在标注框。
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    # 随机的截取图片中一个块。
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4)
    #image = tf.slice(image, bbox_begin, bbox_size)  

    # 标准化处理
    image_as_float = tf.image.convert_image_dtype(image, dtype=tf.float32)
    resized_image = tf.image.resize_images(image_as_float, 
                                           [unified_image_shape[0], unified_image_shape[1]])
    """
    decoded_image_as_float = tf.cast(image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([299, 299])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                                resize_shape_as_int)
    resized_image = tf.subtract(resized_image, 128)
    resized_image = tf.multiply(resized_image, 1.0 / 128)
    resized_image = tf.squeeze(mul_image)
    """
    # 将随机截取的图片调整为神经网络输入层的大小。

    #resized_image = tf.image.random_flip_left_right(mul_image)
    #resized_image = distort_color(mul_image, np.random.randint(2))  #①

    return resized_image

'''
1，当我取消①处的预处理后，图片生成会出问题，怎么回事？
'''