# My first GAN implementation

Here we implement a basic GAN implementation that aims to create fake number images based on the MNIST dataset.


### About explicitly setting the gradients to zero before starting backpropagation.

https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch


Questions

-  2 backward passes for discriminator with fake and real loss or combination of fake an real loss in 1 backward pass?