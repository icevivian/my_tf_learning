import os
import tensorflow as tf
import cProfile

tf.executing_eagerly()  # 开启eager模式,即时处理并返回结果

x = [[2.]]
m = tf.matmul(x,x)
print('hello, {}'.format(m))

a = tf.constant([[1,2],[3,4]])
b = tf.add(a,1)
print(b)
print(a*b) # 点乘，矩阵乘应该用tf.matmul(a,b)

import numpy as np
c = np.multiply(a,b)
print(c)  # 点乘，与上面的等价
print(a.numpy())

def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy()+1):
        num = tf.constant(num)
        if int(num%3) == 0 and int(num%5) == 0:
            print('FizzBuzz')
        elif int(num%3) == 0:
            print('Fizz')
        elif int(num%5) == 0:
            print('Buzz')
        else:
            print(num.numpy())
        counter += 1

# 计算梯度
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w * w
grad = tape.gradient(loss, w)
print(grad)

# train a model
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data() # (60000, 28, 28)

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis]/255, tf.float32),
    tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.shuffle(1000).batch(32)                            # 每个batch32个数，共1875个batch

# build the model
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3,3], activation = 'relu', input_shape = (None, None, 1)),
    tf.keras.layers.Conv2D(16, [3,3], activation = 'relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

# 可以直接看一下模型输出
for images, labels in dataset.take(1):
    print('Logits:', mnist_model(images[0:1].numpy()))

# 确定优化器和损失函数
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
loss_history = []

def train_step(images, labels): # 输入是每个batch内的数据和labels
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
        tf.debugging.assert_equal(logits.shape, (32,10)) # 判定模型输出，32是batch大小，10是输出大小
        loss_value = loss_object(labels, logits)
    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

def train(epochs):
    for epoch in range(epochs):
        for (batch, (image, labels)) in enumerate(dataset):
            train_step(images, labels)
        print('Epoch {} finished'.format(epoch))

train(epochs = 3)

# 打印损失
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')

'''
tf.Variable 对象存储可变的 tf.Tensor， 是训练期间可访问的值，可自动求导
layers 
models
'''
class Linear(tf.keras.Model):
    def __init__(self):
        super(Liner, self).__init__()
        self.W = tf.Variable(5. , name = 'weight')
        self.B = tf.Variable(10. , name = 'bias')
    def call(self, inputs):
        return inputs * self.W + self.B

NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

def loss(model, inputs, targets):
    error = model(inputs)-targets
    return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])

model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
print('Initial loss: {:.3f}'.format(loss(model, training_inputs, training_outputs)))

steps = 300
for i in range(steps):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i % 20 == 0:
        print('Loss at step {:03d}: {:.3f}'.format(i, loss(model, training_inputs, training_outputs)))
print('W = {}, B = {}'.format(model.W.numpy(), model.B.numpy()))
