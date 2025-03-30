import numpy as np
from scipy.ndimage import gaussian_filter


def sinusoidal_encoding(t, d_model):
    """This Function does the positional embedding as described in the "Attention is all you need" paper."
To adapt this to image, d_model can be the square of the side of the image so that the reshaped version
of the encoding vector can be added as input channel.
Pretty sure there's a faster way to do this, but it works.
    """
    encoding = np.zeros(d_model)
    for i in range(0, d_model, 2):
        div_term = 10000 ** (i / d_model)
        encoding[i] = np.sin(t / div_term)
        encoding[i+1] = np.cos(t / div_term)
    return encoding


def add_random_holes(X, p=0.1):
    """This Function adds random 'holes' to the X, with probability (1-p)"""
    mask = np.random.rand(*X.shape) > p 
    return X * mask.astype(X.dtype)


def add_blur(X, sigma=1.0):
    """This Function applies a Gaussian blur to the input array X."""
    return gaussian_filter(X, sigma=sigma)

def timestep_hole(X, t, add_blur=True, hole_per_time=lambda x: 0.1*x, sigma=0.1):
    """This Functions takes X, an image and t, a timestep and adds holes to it.
Then it adds a soft gaussian blur to mimic the effect of convolution networks, and finally
it adds the position encoding channel.

The Probability of holes is computed as a function of the timestep:

X has to be [N, C, W, H]"""
    holed = add_random_holes(X, p=1-hole_per_time(t))
    if add_blur:
        holed = gaussian_filter(holed, sigma=sigma)

    encoded = sinusoidal_encoding(t, X.shape[2]*X.shape[3])
    encoded = encoded.reshape(1, 1, X.shape[2], X.shape[3])
    encoded = np.repeat(encoded, X.shape[0], axis=0)
    return np.concatenate([holed, encoded], axis=1)


def square_hole(X, p, l=1, m=100):
    """
    Generates random squares of zeros inside the given array X with probability p.

    Parameters:
    - X: The input array of shape [N, C, W, H].
    - p: The probability (between 0 and 1) of a square being placed in each channel.

    Returns:
    - An array of the same shape as X, with random squares replaced by zeros.
    """
    N, C, W, H = X.shape
    for n in range(N):
        for c in range(C):
            if np.random.rand() < p:
                square_size = np.random.randint(l, m)
                x_start = np.random.randint(0, W - square_size + 1)
                y_start = np.random.randint(0, H - square_size + 1)
                X[n, c, x_start:x_start + square_size, y_start:y_start + square_size] = 0
    return X


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.random.rand(10, 1, 100, 100)
    X_holed = square_hole(X, 1)
    print(X_holed.shape)
    plt.imshow(X_holed[0, 0, :, :])
    plt.show()
    """plt.subplot(1, 3, 1)
    plt.imshow(X_blurred[0, 0, :, :])
    plt.subplot(1, 3, 2)
    plt.imshow(X_blurred2[0, 0, :, :])
    plt.show()
"""