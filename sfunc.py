import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2

#%%
def natural_string(n: int) -> str:
    """Converts an integer to a 4-digit natural string with zero-padding."""
    return f"{abs(n):04d}"

def discretecolors():
    """Returns a discrete colormap where 0 is transparent."""
    colors = [
        (0, 0, 0, 0),      # Fully transparent for 0
        (0.8, 0.2, 0.2, 1),  # Red
        (0.2, 0.6, 0.2, 1), # Green
        (0.1, 0.2, 0.8, 1), # Blue
        (0.8, 0.6, 0.2, 1) # Yellow
        ]
    return mcolors.ListedColormap(colors)

def count_occurrences(strings):
    counter = Counter(strings)
    return counter

def norm_image(im,Imax=255):
    im2 = im.astype(float) - np.min(im)
    im2 = im2 / np.max(im2)
    im2 = (im2 * Imax).astype(np.short)
    return im2

def clip_values(arr, min_threshold, max_threshold):

    clipped_arr = np.copy(arr)
    clipped_arr[clipped_arr < min_threshold] = 0
    clipped_arr[clipped_arr > max_threshold] = 128

    return clipped_arr

def calculate_mse(image1, image2):
    """Calculate Mean Squared Error between two images."""
    return np.sum((image1 - image2) ** 2) / float(image1.shape[0] * image1.shape[1])

def properties(im):
    print("Min: " + str(np.round(np.min(im))))
    print("Med: " + str(np.round(np.median(im))))
    print("Max: " + str(np.round(np.max(im))))
    print("Mean: " + str(np.round(np.mean(im))))
    print("Sum: " + str(np.sum(im)))
    print("Std: " + str(np.std(im)))
    print("Type: " + str(type(im.ravel()[0])))

