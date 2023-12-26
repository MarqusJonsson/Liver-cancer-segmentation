import random
import numpy as np
# from PIL import Image
from albumentations.core.transforms_interface import ImageOnlyTransform

class LogarithmicMappingTransform(ImageOnlyTransform):
    def __init__(self, constant=0.5, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        scale_factor, log_base = random_point_within_curves()
        self.log_base = log_base
        self.constant = constant
        self.scale_factor = scale_factor

    def apply(self, img, **params):
        # print(img.shape)
        width, height, _ = img.shape
        x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        z = x + 1j * y

        # Add a small constant to avoid large black areas
        z_transformed = np.log(z + self.constant) / np.log(self.log_base)

        # Convert back to image coordinates
        x_transformed = np.real(z_transformed)
        y_transformed = np.imag(z_transformed)

        # Scale the transformed coordinates to better fit the image dimensions
        x_transformed = (x_transformed * self.scale_factor + 1) * (width // 2)
        y_transformed = (y_transformed * self.scale_factor + 1) * (height // 2)

        # Convert the image to a NumPy array
        image_array = np.array(img)

        # Interpolate the transformed coordinates to create the new image
        image_transformed = np.zeros_like(image_array)
        for i in range(height):
            for j in range(width):
                new_x, new_y = int(x_transformed[i, j]), int(y_transformed[i, j])
                if 0 <= new_x < width and 0 <= new_y < height:
                    image_transformed[i, j] = image_array[new_y, new_x]

        return image_transformed
    
def f_upper(x):
    return -1.538*x*x*x + 3.863*x*x - 0.4151*x + 1.131

def f_lower(x):
    return -.007036*x*x*x + 0.3926*x*x + 0.4537*x + 1.081

def random_point_within_curves():
    # Define the range of x-values to generate points
    x_min = 0.1
    x_max = 1.9

    # Generate a random x-value within the specified range
    x = random.uniform(x_min, x_max)

    # Calculate the corresponding y-values for both curves at x
    y_upper = f_upper(x)
    y_lower = f_lower(x)

    # Generate a random y-value between y_lower and y_upper
    y = random.uniform(y_lower, y_upper)
    return x, y