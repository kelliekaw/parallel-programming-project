import csv
import os
import sys
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

# Tesseract installation path (falls back to whatever is on PATH)
WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(WINDOWS_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT

# Stop before processing anything if Tesseract can't run. Otherwise every image
# would fail individually and the CSV would be overwritten with blank rows.
def check_tesseract():
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        raise SystemExit(
            "Tesseract is not installed or not on PATH. "
            "Install it, then run this again."
        )

# Use the working tessdata folder from your test
TESSDATA_DIR = os.path.join(BASE_DIR, "tessdata")
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

# One CPU thread per Tesseract process. Otherwise Tesseract's internal OpenMP
# threads compete with our workers and distort the speedup measurements.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

OCR_CONFIG = "--oem 1 --psm 3 -l eng"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

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

    if not os.path.isdir(image_dir):
        raise SystemExit(f"config.txt: image_dir does not exist: {image_dir}")
    if n_workers < 1:
        raise SystemExit(f"config.txt: n_workers must be at least 1, got {n_workers}")

    return image_dir, n_workers

# Add one result to the shared list
def record_result(results, result_lock, filename, text, processing_time_ms):
    with result_lock:
        results.append({
            "filename": filename,
            "text": text,
            "processing_time_ms": processing_time_ms,
        })

# Load images from disk into the shared queue
def loader(image_dir, n_workers, image_queue, results, result_lock):
    # The stop markers are sent in "finally" so workers always get released,
    # even if the loader fails partway through.
    try:
        filenames = sorted(
            filename for filename in os.listdir(image_dir)
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        )

        for filename in filenames:
            try:
                with Image.open(os.path.join(image_dir, filename)) as img:
                    image = img.convert("RGB")  # Forces the full decode from disk
            except Exception as error:
                # Keep the row so the CSV still accounts for every image
                print(f"Could not load {filename}: {error}")
                record_result(results, result_lock, filename, "", 0.0)
                continue

            # Blocks while the queue is full, which caps memory use
            image_queue.put((filename, image))

        print(f"Loaded {len(filenames)} images.")
    finally:
        for _ in range(n_workers):
            image_queue.put(STOP)  # Signal workers to stop after processing

# Preprocess images
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    return image

# OCR Inference
def run_ocr(image):
    text = pytesseract.image_to_string(image, config=OCR_CONFIG)
    return text.strip()

# Worker threads
def worker(image_queue, results, result_lock, verbose=True):
    while True:
        # Wait for an image or stop marker
        item = image_queue.get()

        if item is STOP:
            return

        filename, image = item
        start_time = time.perf_counter()

        # A bad image must not kill the worker: the worker's stop marker would
        # go unconsumed and the remaining images could stall.
        try:
            text = run_ocr(preprocess_image(image))
        except Exception as error:
            print(f"Could not process {filename}: {error}")
            text = ""

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        record_result(results, result_lock, filename, text, processing_time_ms)

        if verbose:
            print(f"Processed {filename} in {processing_time_ms:.2f} ms. Recognized text: {text}")

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
    # Labels are only used for the accuracy report and are optional
    if not os.path.exists(labels_file):
        return labels
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
    labeled_images = 0

    for r in results:
        filename = r["filename"]
        if filename not in labels:
            print(f"Warning: No label found for {filename}. Skipping accuracy calculation for this file.")
            continue
        labeled_images += 1
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
    word_accuracy = correct_words / labeled_images if labeled_images else 0

    print("\n----- Accuracy Report -----")
    print("Total Characters:", total_char)
    print("Total Errors:", total_err)
    print(f"Character Accuracy: {char_accuracy:.2%}")
    print(f"Word Accuracy: {word_accuracy:.2%}")

    return char_accuracy, word_accuracy

# Run the full pipeline once. Returns the results and the total time in ms.
def run_pipeline(image_dir, n_workers, verbose=True):
    # Shared between the loader and workers. The queue is bounded so the loader
    # can't read the whole dataset into memory ahead of the workers.
    image_queue = queue.Queue(maxsize=n_workers * 2)
    results = []
    result_lock = threading.Lock()

    # Create OCR worker threads
    workers = [
        threading.Thread(
            target=worker,
            args=(image_queue, results, result_lock, verbose),
            name=f"OCR-Worker-{i+1}",
        )
        for i in range(n_workers)
    ]
    loader_thread = threading.Thread(
        target=loader,
        args=(image_dir, n_workers, image_queue, results, result_lock),
        name="Loader",
    )

    # Start total pipeline timer
    pipeline_start_time = time.perf_counter()

    for t in workers:
        t.start()
    loader_thread.start()

    # Each worker exits after taking one stop marker, which the loader queues
    # behind all the images, so these joins return once every image is done.
    loader_thread.join()
    for t in workers:
        t.join()

    # End total pipeline timer
    total_pipeline_time_ms = (time.perf_counter() - pipeline_start_time) * 1000

    return results, total_pipeline_time_ms

def main():
    # OCR text can contain characters the console encoding can't represent
    sys.stdout.reconfigure(errors="backslashreplace")

    # Read config file
    check_tesseract()
    image_dir, n_workers = load_config()

    print("----- Parallel OCR Pipeline -----")
    print(f"Image directory: {image_dir}")
    print(f"Number of OCR workers: {n_workers}")

    # Load ground-truth labels for evaluation
    labels = load_labels(image_dir)

    results, total_pipeline_time_ms = run_pipeline(image_dir, n_workers)

    write_csv(results)

    print("\n----- Pipeline Summary -----")
    print(f"Total Images Processed: {len(results)}")
    print(f"Total Pipeline Time: {total_pipeline_time_ms:.2f} ms")

    if labels:
        calculate_accuracy(results, labels)
    else:
        print("No labels.txt found. Skipping accuracy report.")

if __name__ == "__main__":
    main()