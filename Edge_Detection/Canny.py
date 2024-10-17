import numpy as np
import scipy
from PIL import Image

# Function to read image from file
def read_image(filename):
    img = Image.open(filename).convert('L')
    return np.array(img)

# Function to write image to file
def write_image(data, filename):
    img = Image.fromarray(data)
    img.save(filename)

# Function to apply Gaussian filter
def gaussian_filter(data, sigma):
    size = int(3 * sigma)
    x = np.arange(-size, size + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel[:, None] * kernel[None, :]
    kernel = kernel / np.sum(kernel)
    
    return scipy.signal.convolve2d(data, kernel, mode='same')

# Function to compute gradient magnitude and direction
def compute_gradient(data):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    dx = scipy.signal.convolve2d(data, kernel_x, mode='same')
    dy = scipy.signal.convolve2d(data, kernel_y, mode='same')
    
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    gradient_direction = np.arctan2(dy, dx)
    
    return gradient_magnitude, gradient_direction

# Function to apply non-maximum suppression
def non_maximum_suppression(gradient_magnitude, gradient_direction):
    result = np.copy(gradient_magnitude)
    direction = np.round(gradient_direction * 180 / np.pi / 45) * 45
    
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            angle = direction[i, j]
            
            if angle == 0 or angle == 180 or angle == -180:
                if gradient_magnitude[i, j] < gradient_magnitude[i, j - 1] or \
                   gradient_magnitude[i, j] < gradient_magnitude[i, j + 1]:
                    result[i, j] = 0
            elif angle == 45 or angle == -135:
                if gradient_magnitude[i, j] < gradient_magnitude[i - 1, j - 1] or \
                   gradient_magnitude[i, j] < gradient_magnitude[i + 1, j + 1]:
                    result[i, j] = 0
            elif angle == 90 or angle == -90:
                if gradient_magnitude[i, j] < gradient_magnitude[i - 1, j] or \
                   gradient_magnitude[i, j] < gradient_magnitude[i + 1, j]:
                    result[i, j] = 0
            elif angle == 135 or angle == -45:
                if gradient_magnitude[i, j] < gradient_magnitude[i - 1, j + 1] or \
                   gradient_magnitude[i, j] < gradient_magnitude[i + 1, j - 1]:
                    result[i, j] = 0
    
    return result

# Function to apply double thresholding
def double_thresholding(data, low_threshold, high_threshold):
    result = np.zeros(data.shape)
    
    result[data < low_threshold] = 0
    result[(data >= low_threshold) & (data < high_threshold)] = 128
    result[data >= high_threshold] = 255
    
    return result

# Function to apply hysteresis
def hysteresis(data):
    result = np.copy(data)
    
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            if data[i, j] == 128:
                for k in range(-1, 2):
                    for idx in range(-1, 2):
                        if data[i + k, j + idx] == 255:
                            result[i, j] = 255
                            break
    
    result[result != 255] = 0
    
    return result

# Main function
def canny_edge_detector(filename):
    # Read image from file
    data = read_image(filename)
    
    # Apply Gaussian filter
    data = gaussian_filter(data, 1.4)
    
    # Compute gradient magnitude and direction
    gradient_magnitude, gradient_direction = compute_gradient(data)
    
    # Apply non-maximum suppression
    gradient_magnitude = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # Apply double thresholding
    gradient_magnitude = double_thresholding(gradient_magnitude, 50, 150)
    
    # Apply hysteresis
    gradient_magnitude = hysteresis(gradient_magnitude)
    
    # Write result to file
    write_image(gradient_magnitude.astype(np.uint8), 'result.png')

# Test the Canny edge detector
canny_edge_detector('image.jpg')