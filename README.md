# Face Generation using GAN

## Training Data

* the MNIST database of handwritten digits [link](http://yann.lecun.com/exdb/mnist/)

* images of celebrities [link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) [download](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip)

## Important Aspects:

* There are no Max pooling or average Pooling in the layers in discriminator, which is suggested in the DCGAN paper. (originated called "The All Convolutional Net")

* There is no batch normalization in the first layer of the discriminator, which is suggested in the DCGAN paper.

* The dropout layers can be used for regulization in the discriminator (in this project not recommended).

* The training data (images) must be normalized into `(-1, 1)`, because the activation function `tanh` is used in the generator, which means that the values in each generated image is also `(-1, 1)`.

* The learning rate should not be too large, and should be smaller when the batch_size is smaller.

* Batch size should not be too large, because that the training images should be somehow specified, 
which means the it must have some degree of overfitting, in order to generate more realistic faces.
When it is too general, the generated faces will be somehow unclear.

* Input `z_dim` being the length of the random input vector, should not be too small or too large. When it is too small, the generated images can be not sufficiently detailed. When it is too large, a more complex generator network is required.

## Parameter Studies and Results

### Test (MNIST)

#### Architecture of the neural networks

Generator:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| z vector, size: ?  							| 
|Fully connected layer|out: 3 * 3 * 1024|
|Reshape|out: 3, 3, 1024|
|Normalization||
|leaky RELU||
|Convolution Transpose| 1024 depth, 5 kernel size, 1 stride, valid padding, out: 7, 7, 1024	|
|Normalization| |
|leaky RELU||
|Convolution Transpose| 512 depth, 5 kernel size, 1 stride, same padding, out: 7, 7, 512|
|Normalization| |
|leaky RELU| |
|Convolution Transpose| 256 depth, 5 kernel size, 2 stride, same padding, out: 14, 14, 256	|
|Normalization| |
|leaky RELU||
|Convolution Transpose| 128 depth, 5 kernel size, 1 stride, same padding, out: 14, 14, 128	|
|Normalization| |
|leaky RELU||
|Convolution Transpose| 3 depth, 5 kernel size, 2 stride, same padding, out: 28, 28, 3	|
|tanh||

Descriminator:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 28, 28, 3 RGB image   							| 
| Convolution     	| 64 depth, 5 kernel size, 2 stride, same padding, out: 14, 14, 64	|
| leaky RELU					|												|
| Convolution     	| 128 depth, 5 kernel size, 2 stride, same padding, out: 7, 7, 128	|
|Normalization| |
|leaky RELU||
| Convolution     	| 256 depth, 5 kernel size, 2 stride, same padding, out: 4, 4, 256	|
|Normalization| |
|leaky RELU||
|Flatten|out: 4 * 4 * 256|
|Logit|out: 1|
|Sigmoid||


* Hyperparameters

```
alpha=0.2
z_dim = 128
beta1 = 0.5
```

other important parameters:

```
epoch 1-4: learning rate = 0.0005, batch size = 128
epoch 5-8: learning rate = 0.0002, batch size = 64
epoch 9-10: learning rate = 0.0005, batch size = 128
epoch 11-12: learning rate = 0.0002, batch size = 64
```

* Results

![res9](./pics/res_tst9.png)

![loss9](./pics/loss_tst9.png)

* Comments

1. With larger batch size the losses are smoother. The generated pictures (epoch 10 end) are more general.

2. With smaller batch size the losses are more stochastical. The generated pictures (epoch 8/12 end) have more details.

### Test (CelebA)

The architectures of the neural networks are the same as `Test 9`.

* Hyperparameters

```
alpha=0.2
z_dim = 128
beta1 = 0.2
```

other important parameters:

```
epoch 1: learning rate = 0.0003, batch size = 128
epoch 2: learning rate = 0.0002, batch size = 64
epoch 3: learning rate = 0.0001, batch size = 64
epoch 4: learning rate = 0.0001, batch size = 32
epoch 5: learning rate = 0.00005, batch size = 128
epoch 6: learning rate = 0.00002, batch size = 128
epoch 7: learning rate = 0.00005, batch size = 32
```

* Results

![res10](./pics/res_tst10.png)

![loss10](./pics/loss_tst10.png)

* Comments

1. With larger batch size and smaller learning rate, the losses are smoother. 

2. With smaller batch size and larger learning rate, the losses are more stochastical. 

3. The generated images need be more specific, which means that the batch size and learning rate should not be too large.


### Other tests

#### Test 1

The architecture of the generator is slightly changed as below:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| z vector, size: ?  							| 
|Fully connected layer|out: 3 * 3 * 1024|
|Reshape|out: 3, 3, 1024|
|Normalization||
|leaky RELU||
|Convolution Transpose| 512 depth, 5 kernel size, 1 stride, valid padding, out: 7, 7, 512	|
|Normalization| |
|leaky RELU||
|Convolution Transpose| 256 depth, 5 kernel size, 2 stride, same padding, out: 14, 14, 256	|
|Normalization| |
|leaky RELU||
|Convolution Transpose| 128 depth, 5 kernel size, 2 stride, same padding, out: 28, 28, 128	|
|Normalization| |
|leaky RELU||
|Convolution Transpose| 3 depth, 5 kernel size, 1 stride, same padding, out: 28, 28, 3	|
|tanh||

* Hyperparameter

```
alpha = 0.1  #  for leaky RELU
batch_size = 64
z_dim = 128
learning_rate = 0.0002
beta1 = 0.4  # for adam optimizer
epochs = 2
```

* Result

![res1](./pics/res_tst1.png)

#### Test 2

The network architecture and other parameters are the same with `test 1`
* Hyperparameters

```
batch_size = 32
learning_rate = 0.00005
```

* Result

![res2](./pics/res_tst2.png)

#### Test 3

* Hyperparameters

```
alpha=0.2
batch_size = 64
z_dim = 128
learning_rate = 0.0002
beta1 = 0.1
epochs = 2
```

* Result

![res3](./pics/res_tst3.png)

#### Test 4

* Hyperparameters

```
alpha=0.2
batch_size = 64
z_dim = 128
learning_rate = 0.0001
beta1 = 0.1
epochs = 2
```

* Result

![res4](./pics/res_tst4.png)

#### Test 5

* Hyperparameters
```
alpha=0.2
batch_size = 64
z_dim = 256
learning_rate = 0.0001
beta1 = 0.1
epochs = 2
```

* Result

![res5](./pics/res_tst5.png)

#### Test 6

* Hyperparameters

```
alpha=0.2
batch_size = 64
z_dim = 64
learning_rate = 0.0001
beta1 = 0.1
epochs = 2
```

* Result

![res6](./pics/res_tst6.png)

* Comments

`z_dim` seems to be too small.

#### Test 7

I changed this time the generator a little bit (the last two layers have been changed.). The architecture of the generator is as below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| z vector, size: ?  							| 
|Fully connected layer|out: 3 * 3 * 1024|
|Reshape|out: 3, 3, 1024|
|Normalization||
|leaky RELU||
|Convolution Transpose| 512 depth, 5 kernel size, 1 stride, valid padding, out: 7, 7, 512	|
|Normalization| |
|leaky RELU||
|Convolution Transpose| 256 depth, 5 kernel size, 2 stride, same padding, out: 14, 14, 256	|
|Normalization| |
|leaky RELU||
|Convolution Transpose| 128 depth, 5 kernel size, 1 stride (changed), same padding, out: 14, 14, 128	|
|Normalization| |
|leaky RELU||
|Convolution Transpose| 3 depth, 5 kernel size, 2 stride (changed), same padding, out: 28, 28, 3	|
|tanh||

* Hyperparameters:

```
alpha=0.2
batch_size = 64
z_dim = 128
learning_rate = 0.0001
beta1 = 0.2
epochs = 2
```

* Result

![res7](./pics/res_tst7.png)

![loss7](./pics/loss_tst7.png)

### Test 8

In comparison with `Test 7`, the `learning rate` is changed from `0.0001` into `0.00001` and `batch_size` is changed into `32`.

* Result

![res8](./pics/res_tst8.png)

![loss8](./pics/loss_tst8.png)

* Comments

The learning rate is too small. With only 2 epoches the networks are not converged well.

## References

1. Udacity Deep Learning Foundation Nanodegree