import os
from PIL import Image
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Path to the 'all screenshots' folder
DATASET_DIR = 'all screenshots'

# Containers for data
dimension_counter = Counter()
channel_counter = Counter()
aspect_ratios = []
folderwise_stats = defaultdict(lambda: {'dims': Counter(), 'channels': Counter()})

# Walk through all subfolders and images
for app_name in os.listdir(DATASET_DIR):
    app_folder = os.path.join(DATASET_DIR, app_name)
    if not os.path.isdir(app_folder):
        continue

    for img_file in os.listdir(app_folder):
        img_path = os.path.join(app_folder, img_file)

        try:
            with Image.open(img_path) as img:
                width, height = img.size
                mode = img.mode  # e.g., RGB, RGBA, L, etc.

                dims = (width, height)
                channels = len(img.getbands())

                # Global stats
                dimension_counter[dims] += 1
                channel_counter[channels] += 1
                aspect_ratios.append(round(width / height, 2))

                # Folder-specific stats
                folderwise_stats[app_name]['dims'][dims] += 1
                folderwise_stats[app_name]['channels'][channels] += 1

        except Exception as e:
            print(f"Error loading {img_path}: {e}")

# ---------- Summary Report ---------- #
print("\n📏 IMAGE DIMENSION ANALYSIS")
print("----------------------------------")
print("Most common dimensions:")
for dims, count in dimension_counter.most_common(5):
    print(f"{dims} — {count} images")

print(f"\nUnique sizes found: {len(dimension_counter)}")
print(f"Typical aspect ratios (top 5): {Counter(aspect_ratios).most_common(5)}")

print("\n🎨 COLOR CHANNEL ANALYSIS")
print("----------------------------------")
for ch_count, freq in channel_counter.items():
    print(f"{ch_count} channels — {freq} images")

# ---------- Optional: Plot Histograms ---------- #
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist([w for (w, h) in dimension_counter.keys()], bins=20, color='skyblue', edgecolor='black')
plt.title('Width Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist([h for (w, h) in dimension_counter.keys()], bins=20, color='salmon', edgecolor='black')
plt.title('Height Distribution')
plt.xlabel('Height (pixels)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
