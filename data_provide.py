from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MnistData/', one_hot=True)
print(len(mnist.train.images))