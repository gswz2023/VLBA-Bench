# VLBA-Bench: A Comprehensive Benchmark for Evaluating Backdoor Attacks on Video-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2412.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2412.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![Pytorch 2.0](https://img.shields.io/badge/pytorch-2.0-orange.svg)](https://pytorch.org/)

> **Abstract:**
> Multimodal Large Language Models (MLLMs) have made continuous progress in video language understanding tasks. However, since such models usually rely on large-scale open data for training, even at data poisoning ratios as low as 0.1%–10%, they may still be implanted with backdoors that trigger targeted anomalous outputs during the inference phase.
>
> **VLBA-Bench** is the first systematic evaluation benchmark for backdoor attacks on video language models. We uniformly evaluate four categories of backdoor attack paradigms on three video language datasets (MSVD, MSRVTT, HMDB51) and one long-video language model (MA-LMM). In systematic experiments covering more than 50 configurations, we reveal that:
> * **Time-distributed trigger-based dirty-label video attacks** achieve high stealth while maintaining relatively strong clean performance.
> * **Text-based trigger attacks** can reach high attack success rates under extremely low poisoning ratios but substantially degrade performance on clean samples.
> * **Clean-label attacks and static 2D video triggers** are considerably less effective in poisoning video MLLMs.
>
> This work systematically reveals the modality differences and time-dependent characteristics of backdoor attacks in video-language models.



---

## 🔔 News
* **[2025-12-27]** 🚀 **VLBA-Bench** is released! The code will be fully available soon. Stay tuned!
* **[2025-12-27]** Our paper is available on arXiv.

---

## 🌟 Key Features

* **Systematic Benchmark:** The first unified framework evaluating backdoor robustness specifically for Video-LMMs.
* **Four Attack Paradigms:**
    1.  **Static 2D Attacks:** (BadNets, Blend, SIG, WaNet, FTrojan)
    2.  **Temporal-Distributed Attacks:** Exploiting time-dependent triggers.
    3.  **Clean-Label Attacks:** Including Patch-based and our proposed **Global Perturbation** triggers.
    4.  **Textual Attacks:** Evaluating high-efficiency text injections.
* **Extensive Evaluation:** Covers 50+ configurations across **MSVD**, **MSRVTT**, and **HMDB51** datasets.
* **Modality Analysis:** Provides deep insights into the trade-offs between *Stealthiness*, *Utility*, and *Attack Success Rate* across video and text modalities.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/YourUsername/VLBA-Bench.git](https://github.com/YourUsername/VLBA-Bench.git)
cd VLBA-Bench

# Create environment
conda create -n vlba python=3.9
conda activate vlba

# Install dependencies
pip install -r requirements.txt
```
## 📂 Data Preparation
Please download the datasets and organize them as follows:
data/
├── HMDB51/
│   ├── videos/          # Raw video files
│   └── annotations/     # QA pairs and captions
├── MSVD/
│   ├── videos/
│   └── annotations/
└── MSRVTT/
    ├── videos/
    └── annotations/

## 🚀 Usage
We provide a unified entry point main_attack.py to run evaluations across different paradigms.

### 1. Static 2D Attack (Baseline)
Run classical image backdoors applied frame-by-frame.
```bashhon main_attack.py \
  --method static \
  --trigger_type badnets \
  --dataset msvd \
  --poison_ratio 0.1 \
  --target_word "man"
```
### 2. Temporal-Distributed Attack (High Stealth)
Evaluate the effectiveness of temporally sparse triggers (Section 4.4).

```bash
python main_attack.py \
  --method temporal \
  --dataset msvd \
  --poison_ratio 0.05 \
  --frame_poison_rate 0.5
```

### 3. Text-Based Attack (High Efficiency)
Test the extreme sensitivity of MLLMs to textual triggers (Section 5.1).
```bash
python main_attack.py \
  --method text \
  --poison_ratio 0.0015 \  # Note the extremely low ratio (0.15%)
  --dataset msvd
```
### 4. Clean-Label Attack

Run attacks without modifying the ground-truth labels.

```bash
python main_attack.py \
  --method clean_label \
  --trigger_type global_perturb \
  --budget 16
```

## 📊 Benchmark Results

### Modality Comparison: Text vs. Video
Our experiments reveal a significant **"Data Efficiency Gap"** between modalities.

| Modality | Attack Type | Poison Ratio | ASR (%) | Clean Acc Drop | Characteristics |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Video** | Static 2D (BadNets) | 10.0% | ~50% | High | Requires high poison ratio; naive extension fails. |
| **Video** | **Temporal (TDBA)** | **5.0%** | **~33%** | **Low** | **High Stealth, maintains clean utility.** |
| **Text** | Text Injection | **0.15%** | **~54%** | **High** | **Extremely data-efficient, but destroys utility.** |

### Static 2D Attacks on MSVD (QA Task)
Performance of traditional image backdoors transferred to video.

| Poison Ratio | BadNets | Blend | SIG | FTrojan | WaNet |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 52.72 | 49.98 | 56.53 | 60.24 | 64.43 |
| **5%** | 6.96 | 4.64 | 8.12 | 7.98 | 8.98 |
| **1%** | 0.35 | 1.26 | 2.34 | 0.89 | 2.12 |

> **Observation:** Static 2D triggers largely fail at low poison ratios (1-5%), demonstrating the need for temporal-aware attack strategies in Video-LMMs.

---

## 📝 Citation

If you find **VLBA-Bench** useful for your research, please consider citing our paper:

```bibtex
@article{yourname2025vlba,
  title={VLBA-Bench: A Comprehensive Benchmark for Evaluating Backdoor Attacks on Video-Language Models},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:2412.xxxxx},
  year={2025}
}
