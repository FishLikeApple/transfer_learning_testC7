# -- coding: utf-8 --
"""
此文件实现inception_v3迁移学习模型创建的相关子函数。
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
from tensorflow.python.platform import gfile
import numpy as np

class inference_of_inception_v3_for_fine_tuning:
    BOTTLENECK_OUTPUT_TENSOR_NAME = "pool_3/_reshape:0"
    INPUT_DATA_TENSOR_NAME = "Mul:0"

    #加载模型，定义全连接层和相关步骤
    def __init__(self, number_of_calsses, learning_rate, model_file):
        # 读取训练好的inception_v3模型。
        with gfile.FastGFile(model_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量。
            self.bottleneck_output_tensor, self.input_data_tensor = tf.import_graph_def(
                graph_def, return_elements=[self.BOTTLENECK_OUTPUT_TENSOR_NAME, self.INPUT_DATA_TENSOR_NAME])

        #定义最后的全连接层。
        with tf.name_scope('final_fully_connected_network'):
            FC_input_tensor_size = self.bottleneck_output_tensor.get_shape()[1].value
            self.FC_input = tf.placeholder(tf.float32, [None, FC_input_tensor_size])
            self.FC_labels = tf.placeholder(tf.int64, [None])
            weights = tf.Variable(tf.truncated_normal([FC_input_tensor_size, number_of_calsses], stddev=0.1))
            biases = tf.Variable(tf.zeros([number_of_calsses]))
            logits = tf.matmul(self.FC_input, weights) + biases
            self.final_tensor = tf.nn.softmax(logits)

        #定义损失和训练步骤。
        tf.losses.softmax_cross_entropy(tf.one_hot(self.FC_labels, number_of_calsses), self.final_tensor) 
        self.training_step = tf.train.RMSPropOptimizer(learning_rate).minimize(tf.losses.get_total_loss())  

        #定义准确率和评估步骤。
        correct_prediction = tf.equal(tf.argmax(self.final_tensor, 1), self.FC_labels)
        self.evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def training_op(self, sess, image_input, label_input):
        #首先使用inception_v3模型处理数据
        images =  np.expand_dims(image_input, 1)
        batch_size = image_input.shape[0]
        bottleneck_output = []
        for i in range(batch_size):
            bottleneck_output.append(sess.run(self.bottleneck_output_tensor, 
                                              feed_dict={self.input_data_tensor: images[i]}))
        bottleneck_output = np.squeeze(bottleneck_output)

        #进行训练，返回loss
        _, loss = sess.run([self.training_step, tf.losses.get_total_loss()],
                             feed_dict={self.FC_input: bottleneck_output, self.FC_labels: label_input})
        return loss

    def evaluation_op(self, sess, image_input, label_input):
        #首先使用inception_v3模型处理数据
        images =  np.expand_dims(image_input, 1)
        batch_size = image_input.shape[0]
        bottleneck_output = []
        for i in range(batch_size):
            bottleneck_output.append(sess.run(self.bottleneck_output_tensor, 
                                              feed_dict={self.input_data_tensor: images[i]}))
        bottleneck_output = np.squeeze(bottleneck_output)

        #进行评估，返回准确率
        return sess.run(self.evaluation_step, 
                        feed_dict={self.FC_input: bottleneck_output, self.FC_labels: label_input})

    #输出模型输出的特征向量，输出的文件名为“图像名.txt”。
    def bottleneck_test(self, sess, file_path):
        raw_image = gfile.FastGFile(file_path, "rb").read()
        image = tf.image.decode_jpeg(raw_image)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [299, 299])
        image_values = sess.run(image)        
        expanded_image_values = np.expand_dims(image_values, 0)
        bottleneck_values_1 = sess.run(self.bottleneck_output_tensor, feed_dict={self.input_data_tensor: expanded_image_values})
        bottleneck_values_1 = np.squeeze(bottleneck_values_1)

        expanded_image_values = sess.run(self.decoded_image_tensor,{self.jpeg_data_tensor: raw_image})
        bottleneck_values_2 = sess.run(self.bottleneck_output_tensor, {self.input_data_tensor: expanded_image_values})
        bottleneck_values_2 = np.squeeze(bottleneck_values_2)
        '''
        bottleneck_values_2 = official_functions.run_bottleneck_on_image(sess, raw_image, self.jpeg_data_tensor,
                                                                         self.decoded_image_tensor, self.input_data_tensor,
                                                                         self.bottleneck_output_tensor)
        '''
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(file_path + '.txt', 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)