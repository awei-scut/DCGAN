import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='15'
##载入数据
from tensorflow.examples.tutorials.mnist import input_data

def load_mnist():
    mnist = input_data.read_data_sets('./Mnits_data', one_hot='true')
    #show some images
    for index, image in enumerate(mnist.train.images[:16]):
        image = np.reshape(image, [28, 28])
        plt.subplot(4, 4, index + 1)
        plt.imshow(image, cmap='gray')
        plt.axis("off")
    plt.show()
    return mnist

class GAN:

    def __init__(self):
        with tf.variable_scope('generator'):
            self.G_w1 = tf.Variable(tf.truncated_normal([100, 128], 0.01), name = "G_w1", dtype=tf.float32)
            self.G_b1 = tf.Variable(tf.zeros([128]), name = "G_b1", dtype=tf.float32)
            self.G_w2 = tf.Variable(tf.truncated_normal([128, 784], 0.01), name = "G_w2", dtype=tf.float32)
            self.G_b2 = tf.Variable(tf.zeros([1]), name = "G_b2", dtype=tf.float32)
        with tf.variable_scope('discriminator'):
            self.D_w1 = tf.Variable(tf.truncated_normal([784, 128], 0.01), name = "D_w1", dtype=tf.float32)
            self.D_b1 = tf.Variable(tf.zeros([128]), name = "D_b1", dtype=tf.float32)
            self.D_w2 = tf.Variable(tf.truncated_normal([128, 1], 0.01), name = "D_w2", dtype=tf.float32)
            self.D_b2 = tf.Variable(tf.truncated_normal([1]), name = "D_b2", dtype=tf.float32)


    # model
    def generator(self, inputs):
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_w1) + self.G_b1)
        G_h2 = tf.nn.sigmoid(tf.matmul(G_h1, self.G_w2) + self.G_b2) 
        return G_h2

    def discriminator(self,inputs):

        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_w1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_w2) + self.D_b2
        D_h2 = tf.nn.sigmoid(D_logit)
        return D_h2, D_logit

if __name__ == "__main__":

    # load_mnist()

    mnist = input_data.read_data_sets('./mnits_data', one_hot='true')
    inputs = tf.placeholder(tf.float32, shape=[None, 100])
    gan_mnist = GAN()
    G_output = gan_mnist.generator(inputs)
    X = tf.placeholder(tf.float32, shape=[None, 784])

    D_real, D_logit_real = gan_mnist.discriminator(X)
    D_fake, D_logit_fake = gan_mnist.discriminator(G_output)

    D_loss_real =-tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                labels=tf.ones_like(D_logit_real)))
    D_loss_fake = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    
    train_var = tf.trainable_variables()
    # generator中的tensor
    g_vars = [var for var in train_var if var.name.startswith("generator")]
    # discriminator中的tensor
    d_vars = [var for var in train_var if var.name.startswith("discriminator")]
    D_solver = tf.train.AdamOptimizer(1e-3).minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.AdamOptimizer(1e-3).minimize(G_loss, var_list=g_vars)

    batch_size = 64
    # 训练迭代轮数
    epochs = 300
    # 抽取样本数
    n_sample = 25
    # 存储测试样例
    samples = []
    # 存储loss
    losses = []
    # 保存生成器变量
    saver = tf.train.Saver(var_list = g_vars)
    # 开始训练
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(40):
            for batch_i in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)
                
                batch_images = batch[0].reshape((batch_size, 784))
                # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
                batch_images = batch_images*2 - 1
                
                # generator的输入噪声
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, 100))
                
                # Run optimizers
                _ = sess.run(D_solver, feed_dict={X: batch_images, inputs: batch_noise})
                _ = sess.run(G_solver, feed_dict={inputs: batch_noise})

            # 每一轮结束计算loss
            train_loss_d = sess.run(D_loss, 
                                    feed_dict={X: batch_images, inputs: batch_noise})
            # real img loss
            train_loss_d_real = sess.run(D_loss_real, 
                                        feed_dict={X: batch_images, inputs: batch_noise})
            # fake img loss
            train_loss_d_fake = sess.run(D_loss_fake, 
                                        feed_dict={X: batch_images, inputs: batch_noise})
            # generator loss
            train_loss_g = sess.run(G_loss, 
                                    feed_dict={X: batch_images, inputs: batch_noise})
            
                
            print("Epoch {}/{}...".format(e+1, epochs),
                "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
                "Generator Loss: {:.4f}".format(train_loss_g))    
            # 记录各类loss值
            # losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))
            
            # 抽取样本后期进行观察
                        
    #         # 存储checkpoints
    #         saver.save(sess, './checkpoints/generator.ckpt')

    # # 将sample的生成数据记录下来
    # with open('train_samples.pkl', 'wb') as f:
    #     pickle.dump(samples, f)
