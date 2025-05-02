import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset

dataset = load_dataset("Lexski/sudoku-image-recognition", split="train")

# Use dataset[i]['image'/'cells'/'keypoints'] to access the data
# Graph distribution of digits
solved_digits = np.zeros(10, dtype=int)
small_digits = np.zeros(10, dtype=int)
unsolved_cell_amount = np.zeros(10, dtype=int)
for sudoku in dataset['cells']:
    for row in sudoku:
        for cell in row:
            if cell[0] == 0:
                candidates = np.array(cell[1:])
                valid_digits = candidates == 1
                digit_amount = np.sum(valid_digits)
                small_digits[0] += 1
                small_digits[1:] += valid_digits.astype(int)
                unsolved_cell_amount[digit_amount] += 1
            else:
                solved_digits[0] += 1
                solved_digits[np.argmax(cell[1:]) + 1] += 1

digits = list(range(1, 10))
plt.figure(figsize=(10, 5))
plt.bar(digits, solved_digits[1:], width=0.4, label="Solved", align="center")
plt.bar([d + 0.4 for d in digits], small_digits[1:], width=0.4, label="Unsolved", align="center")
plt.xlabel("Digit")
plt.ylabel("Frequency")
plt.title("Frequency of Each Digit (Solved and Unsolved)")
plt.xticks([d + 0.2 for d in digits], digits)
plt.legend()
plt.tight_layout()
plt.show()

candidate_sizes = list(range(1, 10))
plt.figure(figsize=(10, 5))
plt.bar(candidate_sizes, unsolved_cell_amount[1:])
plt.xlabel("Number of Digits in Unsolved Cell")
plt.ylabel("Frequency")
plt.title("Distribution of Digit Amounts in Unsolved Cells")
plt.xticks(candidate_sizes)
plt.tight_layout()
plt.show()

# Graph image sizes
size_counter = Counter()
for img in dataset['image']:
    size = img.size
    size_counter[size] += 1
widths = [size[0] for size in size_counter]
heights = [size[1] for size in size_counter]
counts = [size_counter[size] for size in size_counter]

plt.figure(figsize=(10, 6))
plt.scatter(widths, heights, s=[c * 2 for c in counts], alpha=0.6)
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Image Size Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph grayscale image pixel intensity
grayscale_images = []
for image in dataset['image']:
    grayscale_image = ImageOps.grayscale(image)
    np_image = np.array(grayscale_image)
    grayscale_images.append(np_image)
all_pixels = np.concatenate([img.flatten() for img in grayscale_images])

plt.hist(all_pixels, bins=256, range=(0, 255), color='red')
plt.title("Histogram of Pixel Intensities for Grayscale Sudokus")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# Visual representation of images
small_images = [img.resize((64, 64), Image.Resampling.LANCZOS) for img in dataset['image']]
n_rows, n_cols = 5, 10
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5), tight_layout=False)
plt.subplots_adjust(wspace=0, hspace=0)
for i, ax in enumerate(axes.flat):
    if i < len(small_images):
        ax.imshow(small_images[i], cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')
plt.show()

# Grayscale image example?