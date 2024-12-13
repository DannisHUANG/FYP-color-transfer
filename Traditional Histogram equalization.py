import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # Normalize CDF for plotting

    # Mask all zeros (to avoid division by zero)
    cdf_m = np.ma.masked_equal(cdf, 0)

    # Normalize the CDF
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Use the CDF to map the original gray levels to equalized levels
    img_equalized = cdf[image]

    return img_equalized

def plot_histograms(original, equalized_images, iterations):
    # Plot original and equalized histograms
    plt.figure(figsize=(18, 12))

    for idx, (equalized, iteration) in enumerate(zip(equalized_images, iterations)):
        plt.subplot(3, 4, idx * 4 + 1)
        plt.title(f"Original Image (Iteration {iteration})")
        plt.imshow(original, cmap='gray')
        plt.axis('off')

        plt.subplot(3, 4, idx * 4 + 2)
        plt.title(f"Equalized Image (Iteration {iteration})")
        plt.imshow(equalized, cmap='gray')
        plt.axis('off')

        plt.subplot(3, 4, idx * 4 + 3)
        plt.title("Original Histogram")
        plt.hist(original.flatten(), 256, [0, 256], color='black')
        plt.xlim([0, 256])

        plt.subplot(3, 4, idx * 4 + 4)
        plt.title("Equalized Histogram")
        plt.hist(equalized.flatten(), 256, [0, 256], color='black')
        plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

def main():
    # Load image
    image_path = r'C:\Users\UIC\Downloads\monai.jpg'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not read the image.")
        return

    # Number of iterations
    num_iterations = 100
    selected_iterations = {1, 10, 100}
    equalized_images = []

    # Iteratively apply histogram equalization
    for i in range(num_iterations):
        equalized_image = histogram_equalization(image)
        if i + 1 in selected_iterations:
            equalized_images.append(equalized_image)
        image = equalized_image  # Use the equalized image for the next iteration

    # Plot results for selected iterations
    plot_histograms(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), equalized_images, selected_iterations)

if __name__ == "__main__":
    main()
