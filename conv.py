import numpy as np


class ConvNet:
    def __init__(self):
        pass

    def conv2d(self, input, kernel, stride=1, padding=0):
        """
        Perform 2D convolution.

        Args:
            input (np.ndarray): Input 2D array.
            kernel (np.ndarray): Kernel 2D array.
            stride (int): Stride for sliding the kernel.
            padding (int): Padding applied to the input array.

        Returns:
            np.ndarray: Output after convolution.
        """
        if padding > 0:
            input = np.pad(
                input, ((padding, padding), (padding, padding)), mode="constant"
            )

        m, n = input.shape
        k, _ = kernel.shape
        output_height = (m - k) // stride + 1
        output_width = (n - k) // stride + 1

        output = np.zeros((output_height, output_width))
        for i in range(0, output_height):
            for j in range(0, output_width):
                region = input[i * stride : i * stride + k, j * stride : j * stride + k]
                output[i, j] = np.sum(region * kernel)

        return output

    def max_pool2d(self, input, pool_size=2, stride=2):
        """
        Perform 2D max pooling.

        Args:
            input (np.ndarray): Input 2D array.
            pool_size (int): Size of the pooling window.
            stride (int): Stride for sliding the pooling window.

        Returns:
            np.ndarray: Output after max pooling.
        """
        m, n = input.shape
        output_height = (m - pool_size) // stride + 1
        output_width = (n - pool_size) // stride + 1

        output = np.zeros((output_height, output_width))
        for i in range(0, output_height):
            for j in range(0, output_width):
                region = input[
                    i * stride : i * stride + pool_size,
                    j * stride : j * stride + pool_size,
                ]
                output[i, j] = np.max(region)

        return output

    def avg_pool2d(self, input, pool_size=2, stride=2):
        """
        Perform 2D average pooling.

        Args:
            input (np.ndarray): Input 2D array.
            pool_size (int): Size of the pooling window.
            stride (int): Stride for sliding the pooling window.

        Returns:
            np.ndarray: Output after average pooling.
        """
        m, n = input.shape
        output_height = (m - pool_size) // stride + 1
        output_width = (n - pool_size) // stride + 1

        output = np.zeros((output_height, output_width))
        for i in range(0, output_height):
            for j in range(0, output_width):
                region = input[
                    i * stride : i * stride + pool_size,
                    j * stride : j * stride + pool_size,
                ]
                output[i, j] = np.mean(region)

        return output

    def relu(self, input):
        """
        Apply ReLU activation function element-wise.

        Args:
            input (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying ReLU.
        """
        return np.maximum(0, input)

    def softmax(self, input):
        """
        Apply softmax activation function.

        Args:
            input (np.ndarray): Input array (1D or 2D).

        Returns:
            np.ndarray: Output after applying softmax.
        """
        exp_input = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return exp_input / np.sum(exp_input, axis=-1, keepdims=True)
