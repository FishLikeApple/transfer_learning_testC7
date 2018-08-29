# transfer_learning_testC7
transfer learning for inception_v3 with some tensorflow manners.

Compared with testC6, the testC7 is implemented with a .pb model file instead of the checkpoint file, and combined with TFrecord files and datasets which the tensorflow recommends.
The accuracy is up to 0.94.

之前使用checkpoint文件的testC6的准确率不是很好，只能到0.65左右。这次的testC7中使用了.pb格式的模型文件，同时增加了TFrecord格式和dataset的相关操作，准确率达到了0.94。
