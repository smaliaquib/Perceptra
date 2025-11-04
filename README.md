# Perceptra

Edge-ready visual anomaly detection—train, evaluate, export, and run lightweight detectors with a clean Python CLI and optional ONNX/C++ runtime.

> **Note:** This repository is part of the broader *AnomaVision* effort (see license header). ([GitHub][1])

## ✨ Features

* **Train models** with a simple entry point: `train.py`. ([GitHub][1])
* **Run inference** on images/folders/streams via `detect.py`. ([GitHub][1])
* **Evaluate** performance with `eval.py` and **benchmark** vs. other libs via `compare_with_anomalib.py`. ([GitHub][1])
* **Export** models for deployment (e.g., ONNX) using `export.py`; optional **C++/ONNX Runtime** integration in `onnxruntime_cpp/`. ([GitHub][1])
* **Configuration-first** via `config.yml`; **usage examples** in `usage_example.py`. ([GitHub][1])
* **Tests** in `tests/`, **code style** via `.pre-commit-config.yaml`. ([GitHub][1])

## 📁 Repository Structure

```
Perceptra/
├─ anodet/                    # Core anomaly detection components
├─ anomavision/               # Shared utilities / higher-level APIs
├─ onnxruntime_cpp/           # Native/C++ runner for ONNX Runtime
├─ tests/                     # Unit/integration tests
├─ app.py                     # (Optional) app entrypoint (API/UI)
├─ train.py                   # Training CLI
├─ detect.py                  # Inference CLI
├─ eval.py                    # Evaluation CLI
├─ export.py                  # Model export CLI (e.g., ONNX)
├─ compare_with_anomalib.py   # Benchmarking script
├─ usage_example.py           # Minimal usage example
├─ config.yml                 # Global configuration
├─ pyproject.toml / poetry.lock
├─ requirements.txt
├─ setup.py / setup.cfg
└─ LICENSE (MIT)
```

(See repo tree for exact files/folders.) ([GitHub][1])

## 🚀 Quickstart

### 1) Environment

Perceptra supports standard Python workflows. Choose **Poetry** or **pip**:

**Poetry**

```bash
# from repo root
poetry install
poetry run python -m pip install --upgrade pip
```

**pip**

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt
```

(Install options reflect the presence of `pyproject.toml`, `poetry.lock`, and `requirements.txt` in the repo.) ([GitHub][1])

### 2) Data

Organize your dataset like:

```
data/
  your_dataset/
    train/
      normal/...
      (optional) masks/...
    test/
      normal/...
      anomalous/...
```

> Adjust paths in `config.yml` or via CLI flags (see below). `config.yml` is present in the repo root. ([GitHub][1])

## 🧠 Training

Basic usage:

```bash
python train.py \
  --data data/your_dataset \
  --config config.yml \
  --project runs/your_experiment
```

Tips:

* Run `python train.py --help` for all flags.
* Use `config.yml` to set model/backbone, image size, augmentations, training hyper-params, and output dirs. ([GitHub][1])

## 🔎 Inference

Single image or a folder:

```bash
python detect.py \
  --weights path/to/best.ckpt_or_onnx \
  --source path/to/image_or_dir \
  --save
```

Options commonly include a confidence/threshold, output directory, and visualization toggles. See `python detect.py --help`. ([GitHub][1])

## 📊 Evaluation

Evaluate on a labeled test set:

```bash
python eval.py \
  --weights path/to/best.ckpt_or_onnx \
  --data data/your_dataset \
  --config config.yml
```

Benchmark against other libraries:

```bash
python compare_with_anomalib.py \
  --data data/your_dataset \
  --report reports/anomalib_comparison.json
```

(Uses the provided comparison script.) ([GitHub][1])

## 📦 Export

Export a trained model for deployment:

```bash
python export.py \
  --weights path/to/best.ckpt \
  --format onnx \
  --out models/exported.onnx
```

ONNX artifacts can be executed with the native runner in `onnxruntime_cpp/`. ([GitHub][1])

## 🧩 Minimal Usage (API)

See `usage_example.py` for a short, end-to-end snippet (load model → run inference → visualize/save results). ([GitHub][1])

## ⚙️ Configuration

* Global defaults live in `config.yml`.
* Most scripts accept `--config` plus CLI overrides, letting you keep reproducible experiments while tweaking individual parameters. ([GitHub][1])

## 🧪 Testing

Run the test suite (examples):

```bash
pytest -q
# or
python -m pytest tests -q
```

(Tests are located in the `tests/` directory.) ([GitHub][1])

## 🧼 Code Style & Hooks

A pre-commit config is included. Enable it to keep formatting, linting, and basic checks consistent:

```bash
pre-commit install
pre-commit run --all-files
```

(See `.pre-commit-config.yaml`.) ([GitHub][1])

## 📝 License

This project is released under the **MIT License**. See `LICENSE`. ([GitHub][1])

## 🙌 Acknowledgements

* The repository includes a `compare_with_anomalib.py` script for benchmarking, which suggests compatibility/contrast with common anomaly-detection baselines. ([GitHub][1])

## 🗺️ Roadmap / TODO

* [ ] Add dataset format docs (CSV/COCO/Mask folder expectations).
* [ ] Publish full CLI reference for `train.py`, `detect.py`, `eval.py`, `export.py`.
* [ ] Provide pretrained weights and example images.
* [ ] Document `onnxruntime_cpp/` build steps and sample commands.
* [ ] Add CI (lint/test) and example GitHub Actions workflow.

---

If you want, I can tailor the README with exact CLI flags (once we lock the argument names) and add a small “Getting Started with an example dataset” section.

[1]: https://github.com/smaliaquib/Perceptra/tree/develop "GitHub - smaliaquib/Perceptra at develop"
