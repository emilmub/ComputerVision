import numpy as np
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
    kernel = np.zeros((2 * size + 1, 2 * size + 1))
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            kernel[i + size, j + size] = np.exp(-(i**2 + j**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    result = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(-size, size + 1):
                for idx in range(-size, size + 1):
                    if 0 <= i + k < data.shape[0] and 0 <= j + idx < data.shape[1]:
                        result[i, j] += kernel[k + size, idx + size] * data[i + k, j + idx]
    return result

# Lagrange interpolation function
def lagrange_interpolation(x, y, x_new):
    result = 0
    for i in range(len(x)):
        p = 1
        for j in range(len(x)):
            if i != j:
                p *= (x_new - x[j]) / (x[i] - x[j])
        result += y[i] * p
    return result

# Numerical differentiation function using Lagrange interpolation
def numerical_differentiation(x, y):
    x_new = (x[0] + x[1]) / 2
    f_x_new = lagrange_interpolation(x, y, x_new)
    h = x[1] - x[0]
    return (y[1] - f_x_new) / (h / 2)

# Function to compute gradient magnitude and direction
def compute_gradient(data):
    gradient_magnitude = np.zeros(data.shape)
    gradient_direction = np.zeros(data.shape)
    
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            x = np.array([j - 1, j, j + 1])
            y = np.array([data[i, j - 1], data[i, j], data[i, j + 1]])
            dx = numerical_differentiation(x, y)
            
            x = np.array([i - 1, i, i + 1])
            y = np.array([data[i - 1, j], data[i, j], data[i + 1, j]])
            dy = numerical_differentiation(x, y)
            
            gradient_magnitude[i, j] = np.sqrt(dx**2 + dy**2)
            gradient_direction[i, j] = np.arctan2(dy, dx)
    
    return gradient_magnitude, gradient_direction

# Function to apply non-maximum suppression
def non_maximum_suppression(gradient_magnitude, gradient_direction):
    result = np.copy(gradient_magnitude)
    
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            angle = gradient_direction[i, j]
            
            if (angle >= -np.pi / 8 and angle < np.pi / 8) or \
               (angle >= 7 * np.pi / 8 and angle < -7 * np.pi / 8):
                if gradient_magnitude[i, j] < gradient_magnitude[i, j - 1] or \
                   gradient_magnitude[i, j] < gradient_magnitude[i, j + 1]:
                    result[i, j] = 0
            elif (angle >= np.pi / 8 and angle < 3 * np.pi / 8) or \
                 (angle >= -7 * np.pi / 8 and angle < -5 * np.pi / 8):
                if gradient_magnitude[i, j] < gradient_magnitude[i - 1, j + 1] or \
                   gradient_magnitude[i, j] < gradient_magnitude[i + 1, j - 1]:
                    result[i, j] = 0
            elif (angle >= 3 * np.pi / 8 and angle < 5 * np.pi / 8) or \
                 (angle >= -5 * np.pi / 8 and angle < -3 * np.pi / 8):
                if gradient_magnitude[i, j] < gradient_magnitude[i - 1, j] or \
                   gradient_magnitude[i, j] < gradient_magnitude[i + 1, j]:
                    result[i, j] = 0
            elif (angle >= 5 * np.pi / 8 and angle < 7 * np.pi / 8) or \
                 (angle >= -3 * np.pi / 8 and angle < -np.pi / 8):
                if gradient_magnitude[i, j] < gradient_magnitude[i - 1, j - 1] or \
                   gradient_magnitude[i, j] < gradient_magnitude[i + 1, j + 1]:
                    result[i, j] = 0
    
    return result

# Function to apply double thresholding
def double_thresholding(data, low_threshold, high_threshold):
    result = np.zeros(data.shape)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] >= high_threshold:
                result[i, j] = 255
            elif data[i, j] >= low_threshold:
                result[i, j] = 128
    
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
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if result[i, j] != 255:
                result[i, j] = 0
    
    return result

# Main function
def canny_edge_detector(filename):
    # Read image from file
    data = read_image(filename)

    write_image(data.astype(np.uint8), 'image_read.png')
    
    # Apply Gaussian filter
    data = gaussian_filter(data, 1.4)

    write_image(data.astype(np.uint8), 'gaussian_applied.png')
    
    # Compute gradient magnitude and direction
    gradient_magnitude, gradient_direction = compute_gradient(data)

    write_image(gradient_magnitude.astype(np.uint8), 'grad_magnit.png')
    write_image(gradient_direction.astype(np.uint8), 'grad_dir.png')
    
    # Apply non-maximum suppression
    gradient_magnitude = non_maximum_suppression(gradient_magnitude, gradient_direction)
    write_image(gradient_magnitude.astype(np.uint8), 'grad_suppr.png')
    
    # Apply double thresholding
    gradient_magnitude = double_thresholding(gradient_magnitude, 1, 10)
    write_image(gradient_magnitude.astype(np.uint8), 'grad_thresh.png')
    
    # Apply hysteresis
    gradient_magnitude = hysteresis(gradient_magnitude)
    
    # Write result to file
    write_image(gradient_magnitude.astype(np.uint8), 'result.png')

# Test the Canny edge detector
canny_edge_detector('image.jpg')