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







if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.random.rand(10, 1, 10, 10)
    X_holed = add_random_holes(X, p=0.00)
    X_blurred = add_blur(X_holed, 0.1)
    X_blurred2 = add_blur(X_holed, 1)

    enc = timestep_hole(X, 3)
    print(enc.shape)
    """plt.subplot(1, 3, 1)
    plt.imshow(X_blurred[0, 0, :, :])
    plt.subplot(1, 3, 2)
    plt.imshow(X_blurred2[0, 0, :, :])
    plt.show()
"""