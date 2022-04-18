import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = io.imread(image_path)
    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    return image[start_row:start_row + num_rows, start_col:start_col + num_cols, :]


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    return image * 0.5 * image * 0.5


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    row_scale_factor = input_rows / output_rows
    col_scale_factor = input_cols / output_cols

    # for i in range(output_rows):
    #     for j in range(output_cols):
    #         output_image[i, j] = input_image[int(i * row_scale_factor), int(j * col_scale_factor)]
    
    # By using this method instead of the nested for loop above
    # runtimes can be slightly improved.
    output_image = [
        input_image[
            int(i * row_scale_factor),
            int(j * col_scale_factor)
        ]
        for i in range(output_rows)
            for j in range(output_cols)
    ]

    # 3. Return the output image
    return np.array(output_image).reshape(output_rows, output_cols, 3)


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!
    
    return np.array([
        point[0] * np.cos(theta) - point[1] * np.sin(theta),
        point[0] * np.sin(theta) + point[1] * np.cos(theta)
    ])


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    # 2. Rotate each pixel of the input image

    for i in range(input_rows):
        for j in range(input_cols):
            point = rotate2d(np.array([i - int(input_rows / 2), j - int(input_cols / 2)]), theta)
            if abs(point[0]) < input_rows / 2 and abs(point[1]) < input_cols / 2:
                output_image[i][j] = input_image[int(input_rows / 2 + point[0]), int(input_cols / 2 - point[1])]

    # 3. Return the output image
    return output_image
