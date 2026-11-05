import csv
import os
import time
import threading
import queue

import pytesseract
from PIL import Image, ImageOps

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ocr_results.csv")

# Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Use the working tessdata folder from your test
TESSDATA_DIR = os.path.join(BASE_DIR, "tessdata")
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

OCR_CONFIG = "--oem 1 --psm 3 -l eng"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Shared variables and locks
image_queue = queue.Queue()
result_lock = threading.Lock()
results = []

# Special marker to signal workers to stop
STOP = None

# Read config file
def load_config():
    config = {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    
    image_dir = config.get("image_dir", "images")
    n_workers = int(config.get("n_workers", 4))

    if not os.path.isabs(image_dir):
        image_dir = os.path.join(BASE_DIR, image_dir)

    return image_dir, n_workers

# Load images into shared queue
def loader(image_dir, n_workers):
    image_paths = []
    for filename in os.listdir(image_dir):
        extension = os.path.splitext(filename)[1].lower()
        if extension in IMAGE_EXTENSIONS:
            image_paths.append(os.path.join(image_dir, filename))

    image_paths.sort()
    for image_path in image_paths:
        image_queue.put(image_path)

    print(f"Loaded {len(image_paths)} images into the queue.")

    for _ in range(n_workers):
        image_queue.put(STOP)  # Signal workers to stop after processing

# Preprocess images
def preprocess_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
    return img

# OCR Inference
def run_ocr(image):
    text = pytesseract.image_to_string(image, config=OCR_CONFIG)
    return text.strip()

# Worker threads
def worker():
    while True:
        # Wait for an image path or stop marker
        item = image_queue.get()

        try:
            if item is STOP:
                return

            image_path = item
            filename = os.path.basename(image_path)

            start_time = time.perf_counter()
            image = preprocess_image(image_path)
            text = run_ocr(image)
            end_time = time.perf_counter()

            processing_time_ms = (end_time - start_time) * 1000

            result = {
                "filename": filename,
                "text": text,
                "processing_time_ms": processing_time_ms,
            }

            with result_lock:
                results.append(result)

            print(f"Processed {filename} in {processing_time_ms:.2f} ms. Recognized text: {text}")

        finally:
            image_queue.task_done()

# Write results to output file
def write_csv(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sorted_results = sorted(results, key=lambda x: x["filename"])
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["ID", "Image Filename", "Extracted Text", "Processing Time (ms)"])

        for idx, result in enumerate(sorted_results, start=1):
            writer.writerow([idx, result["filename"], result["text"], f"{result['processing_time_ms']:.2f}"])

    print(f"Results written to {OUTPUT_FILE}")


# Format: {"filename": "word"}
def load_labels(image_dir):
    labels_file = os.path.join(image_dir, "labels.txt")
    labels = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            word = row["word"]
            labels[filename] = word
    return labels


# Measures how many single-character operations are needed to transform one string into another
# Insertion, deletion, or substitution of a single character
def calculate_edit_distance(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()

    m = len(s1)
    n = len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

def calculate_accuracy(results, labels):
    total_char = 0
    total_err = 0
    correct_words = 0

    for r in results:
        filename = r["filename"]
        if filename not in labels:
            print(f"Warning: No label found for {filename}. Skipping accuracy calculation for this file.")
            continue
        reference = labels[filename]
        recognized = r["text"]

        reference_clean = "".join(
            c for c in reference if c.isalnum()
        )
        recognized_clean = "".join(
            c for c in recognized if c.isalnum()
        )

        errors = calculate_edit_distance(recognized_clean, reference_clean)

        total_char += len(reference_clean)
        total_err += errors

        if recognized_clean.lower() == reference_clean.lower():
            correct_words += 1

    char_accuracy = (total_char - total_err) / total_char if total_char > 0 else 0
    word_accuracy = correct_words / len(results) if results else 0

    print("\n----- Accuracy Report -----")
    print("Total Characters:", total_char)
    print("Total Errors:", total_err)
    print(f"Character Accuracy: {char_accuracy:.2%}")
    print(f"Word Accuracy: {word_accuracy:.2%}")

    return char_accuracy, word_accuracy

def main():
    # Read config file
    image_dir, n_workers = load_config()

    print("----- Parallel OCR Pipeline -----")
    print(f"Image directory: {image_dir}")
    print(f"Number of OCR workers: {n_workers}")

    # Load ground-truth labels for evaluation
    labels = load_labels(image_dir)

    # Create OCR worker threads
    workers = []

    for i in range(n_workers):
        t = threading.Thread(target=worker, name=f"OCR-Worker-{i+1}")
        workers.append(t)

    # Start total pipeline timer
    pipeline_start_time = time.perf_counter()

    # Start worker threads
    for t in workers:
        t.start()

    # Start loader thread
    loader_thread = threading.Thread(target=loader, args=(image_dir, n_workers), name="Loader")
    loader_thread.start()

    # Wait for loader to finish
    loader_thread.join()  

    # Wait for all images to be processed
    image_queue.join()

    # Wait for all worker threads to finish
    for t in workers:
        t.join()

    # End total pipeline timer
    pipeline_end_time = time.perf_counter()

    total_pipeline_time_ms = (pipeline_end_time - pipeline_start_time) * 1000

    write_csv(results)
    char_accuracy, word_accuracy = calculate_accuracy(results, labels)

    print("\n----- Pipeline Summary -----")
    print(f"Total Images Processed: {len(results)}")
    print(f"Total Pipeline Time: {total_pipeline_time_ms:.2f} ms")
    print(f"Character Accuracy: {char_accuracy:.2%}")
    print(f"Word Accuracy: {word_accuracy:.2%}")

if __name__ == "__main__":
    main()