import tensorflow as tf
import pickle
from tensorflow import logging
from tensorflow import gfile
import pprint
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 导入Mnist 数据
from tensorflow.examples.tutorials.mnist import input_data

# os.environ['CUDA_VISIBLE_DEVICES']='15'

# 超参数封装
def get_default_params():
    return tf.contrib.training.HParams(
        z_dim = 100,
        init_conv_size = 4,
        g_channel = [128, 64, 32, 1],
        d_channel = [32, 64, 128, 256],
        learning_rate = 0.002,
        beta1 = 0.5,
        img_size = 32,
        batch_size = 128
    )
  

class MnistData():
    def __init__(self, mnist_train, z_dim, img_size):
        self._data = mnist_train
        self._example_num = len(mnist_train)
        self._z_data = np.random.standard_normal((self._example_num, z_dim))
        self._indicator = 0
        self._random_shuffle()
        self._resize_mnist_img(img_size)

    # 随机shuffle 数据集
    def _random_shuffle(self):
        p = np.random.permutation(self._example_num)
        self._z_data = self._z_data[p]
        self._data = self._data[p]

    # 将图片进行resize
    def _resize_mnist_img(self, img_size):
        data = np.asarray(self._data * 255)
        data = data.reshape((self._example_num, 28, 28))
        new_data = []
        for i in range(self._example_num):
            img = Image.fromarray(data[i])
            img = img.resize((img_size, img_size))
            img = np.asarray(img)
            img = img.reshape((img_size, img_size, 1))
            new_data.append(img)
        new_data = np.asarray(new_data, dtype=np.float32) / 127.5 - 1
        self._data = new_data

    ## next_batch
    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._example_num:
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size

        batch_data = self._data[self._indicator: end_indicator]
        batch_z = self._z_data[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_z

# 反卷积层封装
def conv2d_transpose(inputs, out_channel, name, training, with_relu=True):

    with tf.variable_scope(name):
        conv2d_tran = tf.layers.conv2d_transpose(inputs, out_channel, (5, 5), strides=(2,2), padding="SAME")
        if with_relu == True:
            bn = tf.layers.batch_normalization(conv2d_tran, training=training)
            return tf.nn.relu(bn)
        else:
            return conv2d_tran
# 卷积层封装
def conv2d_warpper(inputs, out_channel, name, training):
    def leaky_relu(x, alpha=0.2, name = ''):
        return tf.maximum(x, alpha * x, name=name)
    with tf.variable_scope(name):
        conv2d = tf.layers.conv2d(inputs, out_channel, [5 ,5], strides=(2, 2), padding="SAME")
        bn = tf.layers.batch_normalization(conv2d, training=training)
        return leaky_relu(bn, name='output')

class Generator():
    def __init__(self, channels, init_conv_size):
        self._channels = channels
        self._init_conv_size = init_conv_size
        self._reuse = False

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('generator', reuse=self._reuse):
            # 先把z_dim 转成初始可反卷积形式
            with tf.variable_scope('inputs_conv'):
                ## reshape 4 * 4 * 128
                fc = tf.layers.dense(inputs, self._channels[0] * self._init_conv_size * self._init_conv_size)
                ## 
                conv0 = tf.reshape(fc, [-1, self._init_conv_size, self._init_conv_size, self._channels[0]])
                bn0 = tf.layers.batch_normalization(conv0, training=training)
                relu0 = tf.nn.relu(bn0)
            deconv_inputs = relu0
            for i in range(1, len(self._channels)):
                wiht_relu = (i != len(self._channels) -1 )
                deconv_inputs = conv2d_transpose(deconv_inputs, self._channels[i], "decov-%d" % i, training, wiht_relu)

        
            img_input = deconv_inputs
            
            with tf.variable_scope('generator_imgs'):
                imgs = tf.tanh(img_input, name='imgs')
            self._reuse = True
            ## 获取变量
            self.variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            return imgs


class Discriminator():
    
    def __init__(self, channels):
        self._channels = channels
        self._reuse = False

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs)

        conv_input = inputs
        with tf.variable_scope('discriminator', reuse=self._reuse):
            for i in range(len(self._channels)):
                print('-----------' + str(i))
                conv_input = conv2d_warpper(conv_input, self._channels[i], 'conv2d_%d' % i, training)

            fc_inputs = conv_input
            with tf.variable_scope('fc'):
                flatten = tf.layers.flatten(fc_inputs)
                logits = tf.layers.dense(flatten,1 , name='logits')
        self._reuse = True
        self.variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return logits

class DCGAN():
    def __init__(self, hps):
        g_channel = hps.g_channel
        d_channel = hps.d_channel

        self._batch_size = hps.batch_size
        self._init_conv_size = hps.init_conv_size
        self._z_dim = hps.z_dim
        self._img_size = hps.img_size

        self._generator = Generator(g_channel, self._init_conv_size)
        self._discriminator = Discriminator(d_channel)

    def build(self):
        
        self._z_placeholder = tf.placeholder(tf.float32, (self._batch_size, self._z_dim))
        self._img_placeholder = tf.placeholder(tf.float32, (self._batch_size, self._img_size, self._img_size, 1))

        generated_img = self._generator(self._z_placeholder, training=True)

        real_logits = self._discriminator(self._img_placeholder, training=True)
        fake_logits = self._discriminator(generated_img, training=True)
        

        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(fake_logits))) \
        + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
        return self._z_placeholder, self._img_placeholder, generated_img, D_loss, G_loss

    def build_op(self, D_loss, G_loss, learning_rate, beta1):
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(G_loss, var_list=self._generator.variable)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(D_loss, var_list=self._discriminator.variable)
        return g_opt, d_opt

        
if __name__ == '__main__':
    # 输出文件夹
    output_dir = './out_run'
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    # 导入数据
    mnist = input_data.read_data_sets('./MnistData/', one_hot=True)
    
    
    data = MnistData(mnist.train.images, 100, 32)
    
    hps = get_default_params()
    dcgan = DCGAN(hps)
    z_placeholder, img_placeholder, generated_img, D_loss, G_loss = dcgan.build()
    g_opt, d_opt = dcgan.build_op(D_loss, G_loss, hps.learning_rate, hps.beta1)
    
    step = 5000
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # 保存生成器变量
    saver = tf.train.Saver(var_list = dcgan._generator.variable)
    # 记录loss
    loss = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(step):
            batch_imgs, batch_z = data.next_batch(hps.batch_size)
            all = [g_opt, d_opt, D_loss, G_loss]
            output_value = sess.run(all, feed_dict={z_placeholder: batch_z, img_placeholder:batch_imgs})
            d_loss_value, g_loss_value = output_value[2:4]
            print("step: %4d, d_loss: %4.3f, g_loss: %4.3f " % (i, d_loss_value, g_loss_value))
            # logging.info("step: %4d, d_loss: %4.3f, g_loss: %4.3f " % (step, d_loss_value, g_loss_value))
            loss.append(output_value[2:4])
            if i % 200 == 0:
                saver.save(sess, './checkpoints/generator.ckpt')
                z_dim = np.random.standard_normal((128, 100))
                sample = sess.run(generated_img, feed_dict={z_placeholder:z_dim})
                
                plt.imshow(sample[20].reshape((32,32)),cmap='gray')
                plt.imshow(sample[21].reshape((32,32)),cmap='gray')
                plt.imshow(sample[22].reshape((32,32)),cmap='gray')
                plt.show()
          

                
                
            

