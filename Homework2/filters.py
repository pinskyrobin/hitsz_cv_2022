import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # This function computes convolution of an image with a kernel and outputs
    # the result that has the same shape as the input image.

    for m in range(Hi):
        for n in range(Wi):
            sum = 0
            for i in range(Hk):
                for j in range(Wk):
                    if m + 1 - i < 0 or \
                        n + 1 - j < 0 or \
                        m + 1 - i >= Hi or \
                        n + 1 - j >= Wi:
                        sum += 0
                    else:
                        sum += kernel[i][j] * image[m + 1 - i][n + 1 - j]
            out[m][n] = sum


    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    
    # out_H = H + 2 * pad_height
    # out_W = W + 2 * pad_width

    # for i in range(out_H):
    #     for j in range(out_W):
    #         if i < pad_height or i >= H + pad_height or j < pad_width or j >= W + pad_width:
    #             continue
    #         else:
    #             out[i][j] = image[i - pad_height][j - pad_width]

    out[pad_height : H + pad_height, pad_width : W + pad_width] = image
    
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image = zero_pad(image, Hk // 2, Wk // 2)
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)
    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum(image[m : m + Hk, n : n + Wk] * kernel)

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g,
    cross correlation is equivalent to a convolution with out flip

    Hint: you can flip `g` at x-axis and y-axis first, 
    and use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    return conv_fast(f, np.flip(np.flip(g, 0), 1))

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    return cross_correlation(f, g - np.mean(g))

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    (you shall not use `conv_fast` above, for you need to normalize each subimage of f)

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    
    Hi, Wi = f.shape
    Hk, Wk = g.shape

    out = np.zeros((Hi, Wi))

    g = (g - np.mean(g)) / np.std(g)

    pad_f = zero_pad(f, int(Hk / 2), int(Wk / 2))

    for i in range(Hi):
        for j in range(Wi):
            _f = pad_f[i : i + Hk, j : j + Wk]
            _f_mean = np.mean(_f)
            _f_std = np.std(_f)
            out[i, j] = np.sum(g * (_f - _f_mean) / _f_std)

    return out
