# VGGNet_TF  
###### 利用Tensorflow简单实现VGGNet，从数据集制作到训练完成测试  
参考：《Tensorflow实战》《Tensorflow 实战Google深度学习框架》

https://blog.csdn.net/sinat_16823063/article/details/53946549

https://blog.csdn.net/yaoqi_isee/article/details/77526497

https://blog.csdn.net/u012759136/article/details/52232266

学习Tensorflow，拿VGG16练练手，没有其他骚操作，只有数据集制作，训练及测试。

训练数据-17flowers，百度网盘链接: https://pan.baidu.com/s/1CXcCgC8Ch5Hdmkgde9yAww 密码: 3nc4

VGG16.npy，百度网盘链接: https://pan.baidu.com/s/1eUlM3ia 密码: 4wvq
* create_tfrecords.py为生成tfrecords数据脚本  
* VGG16.py为网络结构定义文件  
* train.py为训练脚本  
* test.py为测试脚本     

##### 制作tfrecord数据文件  
1. 下载17flowers数据集，解压到目录下  
```
    VGGNet
    |__ 17flowers
        	|__ 0
            	|__ xxx.JPEG
        	|__ 1
        		|__ xxx.JPEG
        	|__ 2
        		|__ xxx.JPEG
```
2. 执行create_tfrecords.py脚本，会在根目录下生成train.tfrecords文件，也可在脚本中指定生成路径    

##### 训练自己的数据  
1. 修改脚本中，模型保存位置及tfrecord数据所在路径，执行train.py脚本即可训练  
2. 训练完成后生成模型文件，执行test.py脚本即可进行测试
3. test文件夹中的图片名字前面数字即为所属类别

网络结构，VGG16.py  
卷积和全连接权重初始化定义了3种方式：  
	1.预训练模型参数  
    2.截尾正态，参考书上采用该方式  
    3.xavier，网上blog有采用该方式  
通过参数finetrun和xavier控制选择哪种方式，有兴趣的可以都试试    
```
def conv(x, d_out, name, fineturn=False, xavier=False):
    d_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # Fine-tuning 
        if fineturn:
            kernel = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print "fineturn"
        elif not xavier:
            kernel = tf.Variable(tf.truncated_normal([3, 3, d_in, d_out], stddev=0.1), name='weights')
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),
                                                trainable=True, 
                                                name='bias')
            print "truncated_normal"
        else:
            kernel = tf.get_variable(scope+'weights', shape=[3, 3, d_in, d_out], 
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),
                                                trainable=True, 
                                                name='bias')
            print "xavier"
        conv = tf.nn.conv2d(x, kernel,[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        print_layer(activation)
        return activation
```  


       训练的时候loss有不收敛的情况，可以适当的调整学习率。

       kee_prob设置为0.5的时候，虽然loss下降到很低，但是测试的效果很差，因为这个纠结了好久。后来改为0.8感觉还可以，可能是因为数据集太少的原因。

       神经网络中的超参数各有各的作用，写完网络在训练的过程中，权重初始化方式，学习率的选择，dropout概率的选择不同都会对训练产生影响，如果想做到指哪打哪还得多积累积累经验。虽然网络可以work，但还是隐隐约约感觉有哪里不对，后期还得优化优化，如有错误欢迎指正交流～



