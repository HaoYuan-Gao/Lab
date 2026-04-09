This is a cudnn using sample, and the cudnn version is v1.18.0. In convolution forward propagation:
$$
Y = X * W
$$

where:

- $X$ is the input tensor  
- $W$ is the convolution kernel  
- $Y$ is the output tensor  

During backpropagation, the gradient with respect to the input is:
$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} * W^T
$$

or equivalently:

$$
dX = dY * W^T
$$

This operation is mathematically equivalent to **transposed convolution**.

Therefore in cuDNN:

- `conv_fprop` → forward convolution  
- `conv_dgrad` → transposed convolution (or input grad)
- `conv_wgrad` → weight gradient