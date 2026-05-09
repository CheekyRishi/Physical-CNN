# 🧠 The Illuminated Neural Network: Physical CNN Hardware Interpreter

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Hardware](https://img.shields.io/badge/Hardware-Arduino_Uno%20%7C%20WS2812B-blue)
![Software](https://img.shields.io/badge/Software-PyTorch%20%7C%20NumPy-orange)
![Award](https://img.shields.io/badge/Award-3rd_Place_Overall-gold)

**Unzipping the "Black Box" of Deep Learning.** This project physically maps the hidden mathematical activations of a Convolutional Neural Network (CNN) into the physical world using an expansive matrix of 1,163 addressable LEDs. By visualizing PyTorch tensors as real-world light, it proves that AI decision-making can be made visible, tangible, and interpretable.

---

## 🏆 Project Highlights
* **3rd Place Overall Winner** out of 20 competing teams (Submission: L2P2-MP03-Group17).
* **Massive Scale:** 1,163 WS2812B LEDs separated across 6 processing stands (Convolutional & Maxpool) and 1 Classification stand.
* **Custom AI Architecture:** A bias-optimized `TinyVGG` model trained to classify Apples, Bananas, and Oranges.
* **Ultra-Fast Streaming:** A custom Python "DJ" script streams pre-calculated tensor data over a 500,000 baud serial connection to the Arduino hardware at 40 FPS.

## 🏗️ System Architecture

Our system bridges high-level deep learning with low-level microcontrollers using a two-stage pipeline.

### 1. The Offline Renderer (PyTorch to NumPy)
Because an Arduino cannot perform thousands of floating-point matrix multiplications per second, we shift the compute load to Python:
* **Forward Hooks:** The script extracts intermediate feature maps (activations) after every Conv2D and MaxPool2D layer.
* **Mathematical Normalization:** Tensor data is normalized to a safe `0-150` brightness range.
* **Tensor Serialization:** The resulting 1,163-element arrays are saved as ultra-compressed `.npy` video files.

### 2. The Visual Enhancer (Sparsity & Compute Shimmer)
To make the "thought process" visible to the human eye, the live streaming script applies real-time Numpy matrix quantization:
* **High Sparsity (Feature Negation):** Low-level noise (values < 40) is ruthlessly clipped to 0, representing the ReLU activation function suppressing useless background data.
* **Compute Shimmer:** Active pixels receive a high-speed randomized noise (+/- 25 brightness) to simulate the live, high-energy dot products occurring during feature extraction.

### 3. The Dual-Pin Hardware Bypass
Due to a 3-meter physical gap between Conv Layer 4 and Maxpool Layer 2, the 800kHz WS2812B square-wave signal experienced severe degradation (capacitance and resistance loss). We engineered a **Dual-Pin Architecture** on the Arduino to fix this:
* **Pin 5:** Drives the first 1,120 LEDs (Conv 1 ➔ Conv 4).
* **Pin 6:** Injects a fresh, uncorrupted 5V TTL signal directly into the final 43 LEDs across the gap.

## 📂 Repository Structure

```text
├── model/
│   ├── tiny_vgg_fruits.pth       # Saved model weights
│   └── model_architecture.py     # PyTorch TinyVGG definition
├── tensors/
│   ├── anim_apple.npy            # Pre-rendered activation video for Apple
│   ├── anim_banana.npy           # Pre-rendered activation video for Banana
│   └── anim_orange.npy           # Pre-rendered activation video for Orange
├── src/
│   ├── generator.py              # Script to generate .npy files via PyTorch hooks
│   ├── live_demo.py              # The interactive Jury CLI and Serial streamer
│   └── dual_pin_test.ino         # Arduino C++ driver with bit-banging logic
└── README.md
