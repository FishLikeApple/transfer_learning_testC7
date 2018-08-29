import tensorflow as tf
import JPG_to_TFRecord
import preprocessing_for_training
import inference
import gc
import scipy.misc

unified_image_shape = [299, 299, 3]
bottleneck_tensor_size = 2048
batch_size = 100
shuffle_buffer = 1000
num_of_epochs = 20
learning_rate = 0.01
number_of_calsses = 5
training_files = tf.train.match_filenames_once(
    "D:/Backup/Documents/Visual Studio 2015/Projects/testC7_1/testC7_1/data/training_data_*")
test_files = tf.train.match_filenames_once(
    "D:/Backup/Documents/Visual Studio 2015/Projects/testC7_1/testC7_1/data/test_data_*")
model_file = "D:/Backup/Documents/Visual Studio 2015/Projects/inception_v3_TL/inception_v3_TL/model/tensorflow_inception_graph.pb"

#解析器。与书上的解析器相比，这里的解析器被做了一点修改。
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            "image":tf.FixedLenFeature([], tf.string),
            "label":tf.FixedLenFeature([], tf.int64),
            "shape":tf.FixedLenFeature([3], tf.int64),
            })
    decoded_image = tf.decode_raw(features["image"],tf.uint8)
    decoded_image = tf.reshape(decoded_image, features["shape"])  #①
    return decoded_image, features["label"] 

def preprocesser(image, label):
    return preprocessing_for_training.preprocessing_for_training(image, unified_image_shape, None), label

def basic_preprocesser(image, label):
    image_as_float = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return tf.image.resize_images(image_as_float, [unified_image_shape[0], unified_image_shape[1]]), label

def main(_):
    #JPG_to_TFRecord.JPG_to_TFRecord("D:/Backup/Documents/Visual Studio 2015/Projects/testC6/testC6/flower_photos")

    #接下来是dataset的一连串映射
    dataset = tf.data.TFRecordDataset(training_files)
    dataset = dataset.map(parser)
    dataset = dataset.map(preprocesser)

    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)
    dataset = dataset.repeat(num_of_epochs)

    iterator_for_training = dataset.make_initializable_iterator()
    image_batch_for_training, label_batch_for_training = iterator_for_training.get_next()

    #测试部分的映射
    dataset = tf.data.TFRecordDataset(test_files)
    dataset = dataset.map(parser)
    dataset = dataset.map(basic_preprocesser)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(99999999999)

    iterator_for_test = dataset.make_initializable_iterator()
    image_batch_for_test, label_batch_for_test = iterator_for_test.get_next()

    #实例化inference中的类
    model_inference = inference.inference_of_inception_v3_for_fine_tuning(number_of_calsses,learning_rate, model_file)

    #创建对话，并且进行相关初始化
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 
    sess.run(tf.local_variables_initializer())                                                                                                 
    sess.run(iterator_for_training.initializer)
    sess.run(iterator_for_test.initializer)

    #开始训练
    i = 0
    while True:
        try:
            images, labels = sess.run([image_batch_for_training, label_batch_for_training])
            loss = model_inference.training_op(sess, images, labels)
            i += 1
            print("loss=%f  i=%d" %(loss,i))
        except tf.errors.OutOfRangeError:
            break
        #每训练10次计算1次准确率。
        if (i%10) == 0:
            images, labels = sess.run([image_batch_for_test, label_batch_for_test])
            print("accuracy=%f" %(model_inference.evaluation_op(sess, images, labels)))
            #如果有需要，可以输出待训练图像以对其进行检验。
            #for k in range(100):
                #scipy.misc.imsave('D:/Backup/Documents/Visual Studio 2015/Projects/testC7_1/testC7_1/data/reimg%d,%d.png'%(i ,labels[k]), images[k])

    #最终测试部分
    images, labels = sess.run([image_batch_for_test, label_batch_for_test])
    print("final accuracy 1=%f" %(model_inference.evaluation_op(sess, images, labels)))
    images, labels = sess.run([image_batch_for_test, label_batch_for_test])
    print("final accuracy 2=%f" %(model_inference.evaluation_op(sess, images, labels)))
    images, labels = sess.run([image_batch_for_test, label_batch_for_test])
    print("final accuracy 3=%f" %(model_inference.evaluation_op(sess, images, labels)))

main(0)
"""
1，①处书上有个错误：
    (1)，使用set_shape()会报错，说是int()的参数不能是tensor。因为我无法定位到它的定义，所以干脆就换成了reshape()。
    参考了：https://blog.csdn.net/qq_21949357/article/details/77987928；
    (2)，features["hight"]，features["width"]和features["num_of_channels"]都是张量，而reshape()的输入则只需要一个张量。
    (3)，如果直接用中括号括起来会报错，
    (4)，可以使用tf.stack()，但是还会报错，说它不是可迭代的（iterable）。借此我也了解到了map_fn()。
    参考：https://blog.csdn.net/loseinvain/article/details/78815130；

"""