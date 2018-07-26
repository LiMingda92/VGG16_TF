#coding=utf-8

import os
import tensorflow as tf
from PIL import Image
import sys

def creat_tf(imgpath):

    cwd = os.getcwd()
    classes = os.listdir(cwd + imgpath)

    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + imgpath + name + "/"
        print class_path
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((224, 224))
                img_raw = img.tobytes()              #将图片转化为原生bytes
                example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
                writer.write(example.SerializeToString())  #序列化为字符串
                print(img_name)
    writer.close()

def read_example():

    #简单的读取例子：
    for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
    
        #image = example.features.feature['img_raw'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        # 可以做一些预处理之类的
        print label

if __name__ == '__main__':
    imgpath = './17flowers/'
    creat_tf(imgpath)
    #read_example()
    

    
