# Import necessary libraries for image processing, data handling, and evaluation
import scipy.io as sio
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import exposure, morphology, filters, measure
from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd

# ============== Load MRI Data ==============

# Get the directory of the current script
project_dir = Path(__file__).resolve().parent

# Create the path to the Brain.mat file (assumed to be in a folder named 'Data')
data_path = project_dir / "Data" / "Brain.mat"

# Load the .mat file using scipy
mat_data = sio.loadmat(data_path)
T1_slices = mat_data['T1']  # MRI slices
label_data = mat_data['label']  # Ground truth segmentation labels
num_slices = T1_slices.shape[2]  # Total number of slices in the volume

# ============== Define Preprocessing and Helper Functions ==============

def preprocess_image(image):
    # Rescale intensity to full range, then smooth with a Gaussian filter
    normalized_image = exposure.rescale_intensity(image, in_range='image')
    smoothed_image = gaussian_filter(normalized_image, sigma=1)
    return smoothed_image

def remove_large_objects(binary_mask, max_size):
    # Label connected components
    labeled_mask = measure.label(binary_mask)
    # Calculate size of each component
    sizes = np.bincount(labeled_mask.ravel())
    # Keep only components smaller than max_size
    filtered_mask = np.isin(labeled_mask, np.where(sizes <= max_size)[0])
    return filtered_mask

def apply_morphological_operation(image, max_size=50000):
    # Use Otsu's method to binarize image
    threshold_value = filters.threshold_otsu(image)
    binary_mask = image > threshold_value
    # Remove small and large objects
    processed_mask = morphology.remove_small_objects(binary_mask, min_size=5000)
    processed_mask = remove_large_objects(processed_mask, max_size)
    return processed_mask

def binary_classification(user_min, user_max, image):
    # Return a binary mask where values within range [min, max] are 0, others 1
    binary_image = np.where((image >= user_min) & (image <= user_max), 0, 1)
    return binary_image, user_min, user_max

def evaluate_segmentation(segmented_image, ground_truth):
    # Flatten arrays for metric calculation
    segmented_flat = segmented_image.flatten()
    ground_truth_flat = ground_truth.flatten()
    # Calculate Jaccard Index and F1 Score
    jaccard = jaccard_score(ground_truth_flat, segmented_flat, average='weighted')
    f1 = f1_score(ground_truth_flat, segmented_flat, average='weighted')
    return jaccard, f1

# ============== Segment and Evaluate Each Slice ==============

results = []             # Store metrics for each slice
segmented_slices = []    # Store segmented image and ground truth for plotting

for slice_index in range(num_slices):
    # Extract one MRI slice and its corresponding label
    slice_data = T1_slices[:, :, slice_index]
    ground_truth_slice = label_data[:, :, slice_index]

    # Normalize slice to 8-bit image
    image_normalize = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Preprocess: intensity normalization + smoothing
    preprocessed_image = preprocess_image(image_normalize)

    # Perform Otsu thresholding and remove unwanted components
    morph_image = apply_morphological_operation(preprocessed_image)

    # Dilate the region to include surrounding pixels
    expanded_mask = binary_dilation(morph_image, iterations=3)

    # Highlight the dilated mask region with intensity 15
    modified_image = np.where(expanded_mask, 15, image_normalize)

    # ===== Region-specific segmentation with value re-encoding =====

    # 1. Air region (intensity = 0)
    segments = {'Air': binary_classification(0, 0, modified_image)}
    air_segment, _, _ = segments['Air']

    # Remove air region from original image
    modified_original_image = np.where(air_segment == 0, 0, modified_image)

    # Reassign the skull/scalp mask to intensity 5
    modified_original_image = np.where(morph_image, 5, modified_original_image)

    # Remove low-intensity noise
    modified_original_image = np.where((modified_image >= 0) & (modified_image <= 13), 0, modified_original_image)

    # 2. Region between intensity 1 and 10
    binary_segment = binary_classification(1, 10, modified_original_image)
    modified_original_image_2 = np.where(binary_segment[0] == 0, 18, modified_original_image)

    # 3. Region between 19 and 40, then close holes
    binary_mask = np.where((modified_original_image_2 >= 19) & (modified_original_image_2 <= 40), 1, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    modified_original_image_3 = np.where(closed == 1, 35, modified_original_image_2)

    # 4. Region between 36 and 89, morphologically refine
    binary_segment_4 = binary_classification(36, 89, modified_original_image_3)
    binary_morphological = apply_morphological_operation(binary_segment_4[0])
    modified_original_image_4 = np.where(binary_morphological == 1, 80, modified_original_image_3)

    # 5. Region between 81 and 130
    binary_segment_5 = binary_classification(81, 130, modified_original_image_4)
    modified_original_image_5 = np.where(binary_segment_5[0] == 0, 120, modified_original_image_4)

    # 6. Region between 121 and 255
    binary_segment_6 = binary_classification(121, 255, modified_original_image_5)
    modified_original_image_6 = np.where(binary_segment_6[0] == 0, 160, modified_original_image_5)

    # ===== Assign final label values (0–5) to different regions =====
    final_segmented_image = modified_original_image_6.copy()
    final_segmented_image[final_segmented_image == 0] = 0   # Air
    final_segmented_image[final_segmented_image == 18] = 1  # Skin
    final_segmented_image[final_segmented_image == 35] = 2  # Skull
    final_segmented_image[final_segmented_image == 80] = 3  # CSF
    final_segmented_image[final_segmented_image == 120] = 4 # Gray Matter
    final_segmented_image[final_segmented_image == 160] = 5 # White Matter

    # Use a binary mask to fill holes and refine air/background
    air_section = binary_classification(0, 60, image_normalize)
    filled_mask = binary_fill_holes(air_section[0])
    final_segmented_image = np.where(filled_mask == 0, 0, final_segmented_image)

    # Ensure only valid labels remain (any other value gets assigned label 1)
    valid_labels = {0, 1, 2, 3, 4, 5}
    final_segmented_image = np.where(np.isin(final_segmented_image, list(valid_labels)), final_segmented_image, 1)

    # Save segmented slice for visualization later
    segmented_slices.append((final_segmented_image, ground_truth_slice, slice_index))

    # Evaluate segmentation quality
    jaccard_index, f1_score_value = evaluate_segmentation(final_segmented_image, ground_truth_slice)
    results.append({'Slice': slice_index, 'Jaccard': jaccard_index, 'F1 Score': f1_score_value})

# ============== Display and Print Evaluation Results ==============

df = pd.DataFrame(results)
print(df)

# Print average scores across all slices
average_jaccard = df['Jaccard'].mean()
average_f1 = df['F1 Score'].mean()
print(f"\nAverage Jaccard Index: {average_jaccard:.4f}")
print(f"Average F1 Score: {average_f1:.4f}")

# ============== Plot Segmented vs Ground Truth Slices ==============

for segmented, ground_truth, idx in segmented_slices:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot segmented result
    im1 = axes[0].imshow(segmented, cmap='jet')
    axes[0].set_title(f'Slice {idx} - Segmentation')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], orientation='horizontal', shrink=0.8, pad=0.05)
    cbar1.set_ticks(np.arange(6))

    # Plot ground truth label
    im2 = axes[1].imshow(ground_truth, cmap='jet')
    axes[1].set_title(f'Slice {idx} - Ground Truth')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], orientation='horizontal', shrink=0.8, pad=0.05)
    cbar2.set_ticks(np.arange(6))

    plt.tight_layout()
    plt.show()
