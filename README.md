# My first GAN implementation

Here we implement a basic GAN implementation that aims to create fake number images based on the MNIST dataset.

GANs belong to the set of algorithms named generative models.
These algorithms belong to the field of unsupervised learning, a sub-set of Machine Learning which aims to study algorithms that learn the underlying structure of given data, without specifying a target value.

## Tips

### Losses

#### Discriminator loss

max [ log(D(real)) + log(1 - D(G(z)) ]

#### Generator loss

min [ log(1 - D(G(z))) ] <-> max [ log(D(G(z))) ]

In these losses the logarithm of the probability is used in the loss function instead of raw probabilities, since using a log loss heavily penalises classifiers that are confident about an incorrect classification.

### Droput layers

It is used to prevent over-fitting.

During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.

This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the paper Improving neural networks by preventing co-adaptation of feature detectors .

## loss.backward()

The backward pass consists of computing the local gradients and dLoss/dWeights using the chain rule to backpropagate the error.

### Load Tensorboard in google colab

```python
# Load the TensorBoard notebook extension.
%load_ext tensorboard
```

```python
# Opens tensorboard with "runs" as the directory where the images are stored.
%tensorboard --logdir runs
```

### About explicitly setting the gradients to zero before starting backpropagation.

https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch


### GAN hacks

https://github.com/soumith/ganhacks

### Questions

-  2 backward passes for discriminator with fake and real loss or combination of fake an real loss in 1 backward pass?