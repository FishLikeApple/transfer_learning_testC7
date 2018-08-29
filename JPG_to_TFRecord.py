# -- coding: utf-8 --
'''
此文件实现JPG到TFRecord的转换，其中对每个JPG源文件都会生成一个TFRecord目标文件。
文件名“test_data_28.TFRecord”表示测试数据的第29个文件（第一个文件的文件号为0）。
'''
import glob
import os.path
import tensorflow as tf
import numpy as np
import gc
from tensorflow.python.platform import gfile

#下函数会由JPG文件得到npy文件
def JPG_to_TFRecord(input_JPG_path, output_file_path = "data", validation_data_ratio = 0.1, 
               test_data_ratio = 0.1):
    file_list = []
    file_labels = []

    #获取所有文件和其标签
    sub_dirs = [x[0] for x in os.walk(input_JPG_path)]  #获取input_JPG_path下的所有子目录名
    extensions = ["jpg", "jpeg"]
    current_label = 0
    for sub_dir in sub_dirs:
        if sub_dir == input_JPG_path: continue
        for extension in extensions:
            file_glob = glob.glob(sub_dir+"/*."+extension)  #不分大小写
            file_list.extend(file_glob)   #添加文件路径到file_list
            file_labels.extend(np.ones(np.shape(file_glob), dtype="int64")*current_label)   #添加标签到file_labels，标签与文件路径数量相同
        current_label +=1

    state = np.random.get_state()
    np.random.shuffle(file_list)
    np.random.set_state(state)
    np.random.shuffle(file_labels)

    traning_count = 0
    test_count = 0
    validation_count = 0
    iteration_times = 0
    sess = tf.Session()   #获取图片数据时会用到
    for file_name in file_list:
        print("label=" + str(file_labels[iteration_times]) + "  file_path=" + file_name)   #打印当前储存的文件和标签
        image_values = tf.image.decode_jpeg(gfile.FastGFile(file_name, "rb").read())
        image_values = sess.run(image_values)

        chance = np.random.random_sample()
        if chance < validation_data_ratio:
            writer = tf.python_io.TFRecordWriter(output_file_path+"/validation_data_"+str(validation_count)+".TFRecord") 
            validation_count += 1
        elif chance < (validation_data_ratio + test_data_ratio):
            writer = tf.python_io.TFRecordWriter(output_file_path+"/test_data_"+str(test_count)+".TFRecord")
            test_count += 1
        else:
            writer = tf.python_io.TFRecordWriter(output_file_path+"/training_data_"+str(traning_count)+".TFRecord") 
            traning_count += 1

        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_values.tostring()])),  #①
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[file_labels[iteration_times]])),
            "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=image_values.shape)),
            }))
        writer.write(example.SerializeToString())
        iteration_times += 1
        gc.collect()

    sess.close()
'''
1，如果在①处
'''