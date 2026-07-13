# EMG-TinyML on ESP32-S3 — B.Sc. Thesis Code

Benchmarking framework for running quantized EMG regression models (Conv + LSTM pipeline, TensorFlow Lite Micro) on the ESP32-S3. The project has two sides:

1. **EMG-TinyML (host / Python)** — converts PyTorch models to `.tflite`, packages them into flashable `.bin` binaries with matching `.h` manifests, and generates test data.
2. **ESP32-S3 firmware (ESP-IDF / C++)** — flashes the models and data, runs inference window-by-window, and emits CSV benchmark reports (latency, MSE/MAE/NMSE, model size, arena usage).

---

## Prerequisites

### ESP32-S3 side
- **ESP-IDF v6.0** is required. Install it either through the [VS Code ESP-IDF extension](https://github.com/espressif/vscode-esp-idf-extension) or the command-line interface (`idf.py`).
- Verify the installation by running `idf.py build` inside the project folder — you should see a successful build.

### Host (conversion) side
- A **conda environment** with `litert_torch` installed, plus miscellaneous libraries: `pandas`, `pyyaml`, `matplotlib`, `numpy`, `torch`, etc.

---

## Workflow 1 — Core EMG Benchmark

### 1. Configure and convert models (host)
1. Open `model_factory.py` and edit the **`Parameter_Config`** dict to specify the model configurations you want to test (input shape, window length, step size, ...).
2. In `config.py`, set where output files are saved and which original model YAML config / weights are used.
3. Activate your conda environment and run **`model_litert.py`** to start the conversion. The number of trials used for quantization calibration is controlled by **`MAX_CALIBRATED_TRIALS`**.
4. Run **`pack_final_models.py`** to package the converted models (from the folder specified in `config.py`) into a single `models_0.bin` plus `.h` manifest files that index the models on the ESP32-S3.
5. Run **`model_sample_gen_trial.py`** to generate the test data — the x/y inputs and expected outputs of the specified trial — as a `.bin` + `.h` pair.

### 2. Transfer artifacts to the firmware project
You now have **4 files**: a `.bin` + `.h` for the models and a `.bin` + `.h` for the data.
- Copy the **binary files** into `data_binaries/`.
- Copy the **header files** into `binary_manifests/`.

### 3. Wire up the firmware
- In `main.cpp`, define `EMG_MODEL_BENCHMARK` and point the `#include`s under that section to your new manifest headers.
- In `CMakeLists.txt`, register the binaries with `esptool_py_flash_to_partition()` for the correct partitions (`benchmark_models`, `benchmark_data`).

> ⚠️ **Consistency matters.** Any mismatch between the headers, binaries, and partitions causes unpredictable behaviour — or a flash error if a `.bin` exceeds its allocated partition size.

### 4. Partition table
Specify an appropriate **`partitions.csv`** that tells the board how flash is divided. Two main partitions are used: one for the **models** and one for the **data**. Size them according to your `.bin` files.

### 5. Hardware / SDK configuration
ESP32-S3 hardware settings (CPU frequency, PSRAM, cache, etc.) can be adjusted via `sdkconfig` directly or interactively with:

```bash
idf.py menuconfig
```

### 6. Quantization configuration
The EMG benchmark tests two variants: **float32** and **int8**. For each model configuration there are **8 models total** (4 float32 + 4 int8 stages: firstA, firstB, LSTM, regressor).

In `run_app()` (main.cpp), pick the flash indices (`idx_a`, `idx_b`, `idx_lstm`, `idx_reg`) and set up the matching `MasterHandle::QuantConfig` (function pointers + labels).

> ⚠️ The chosen model indices and the `QuantConfig` **must be consistent** — a mismatch causes unpredictable behaviour.

---

## Workflow 2 — Random Weights Benchmark

Used for isolated layer-type sweeps (Dense / Conv2D / LSTM) with deterministic random weights. The relevant code lives in the **`brute/`** folder and mirrors the core structure:

1. Specify the models and configurations to test in each section of the brute `Parameter_Config` (e.g. `brute_models.py`).
2. Export them as LiteRT models with **`brute_litert.py`**.
3. Package them into a `.bin` with **`pack_models.py`** (same directory as `pack_final_models.py`).
4. Export random input/output test data with **`brute_sample_gen.py`**.

On the firmware side, define `RANDOM_WEIGHTS_BENCHMARK`, point the includes to the generated manifests, and register the binaries in `CMakeLists.txt` — same as the core experiment.

---

## Workflow 3 — Quantization Fidelity Experiments

Nearly identical to the core workflow, with two differences:

- Use the **multiple-trials sample generator** instead of the single-trial one, so the same fixed model config is evaluated across many trials.
- Adjust **`partitions.csv`** so the data partition is large enough to hold **all trials**.

On the firmware side, define `QUANTIZATION_FIDELITY` and include the corresponding manifests.

---

## Troubleshooting

**Large experiments fail to run** (very large models, or large input/output sizes): reduce the number of models tested per run. This counteracts **heap fragmentation** and the general memory constraints of the ESP32-S3 (limited internal SRAM, PSRAM bandwidth, and TFLite Micro arena requirements).

Other tips:
- Watch the boot log line `FREE_HEAP_CONT_START` for the largest contiguous free block.
- Flash errors during `idf.py flash` usually mean a `.bin` exceeds its partition — regenerate `partitions.csv`.
- The final benchmark output is printed as a single CSV block (`===BENCHMARK_CSV_BEGIN=== ... ===BENCHMARK_CSV_END===` or the `FINAL REPORT` log) that can be captured over serial and parsed directly with pandas.
