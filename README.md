# 🧠 GaussTrap: Stealthy Backdoor Attacks on 3D Gaussian Splatting for Targeted Scene Misperception

> **Accepted to**: ACM SIGKDD Conference on Knowledge Discovery and Data Mining 2026, KDD 2026
>
> **Published**: April 2025  
>
> **Paper Link**: [arXiv:2504.20829](https://arxiv.org/abs/2504.20829)   

---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Installation Guide](#installation-guide)
- [Dataset Description](#dataset-description)
- [Training and Evaluation](#training-and-evaluation)
- [Citation](#citation)
- [License](#license)

---

## 🔍 Introduction

This work introduces **GaussTrap**, a stealthy poisoning attack method targeting **3D Gaussian Splatting (3DGS)** models. By injecting carefully crafted adversarial samples into the training data, this attack induces the model to generate incorrect rendering results from specific viewpoints — achieving "targeted scene confusion" — while maintaining visual quality from other perspectives, thereby ensuring high stealthiness.

---

## ⚙️ Installation Guide

This project uses code from the Gaussian Splatting software, which is available under a custom non-commercial license by Inria and MPII. Please refer to the official implementation of 3D Gaussian Splatting for environment setup:
🔗 [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

We recommend using a Python virtual environment to avoid dependency conflicts.

---

## 📁 Dataset Description

The supported datasets include **Blender** and **MipNeRF-360**.

### Download Links

- **Blender Dataset**: [https://www.matthewtancik.com/nerf](https://www.matthewtancik.com/nerf)
- **MipNeRF-360 Dataset**: [https://jonbarron.info/mipnerf360/](https://jonbarron.info/mipnerf360/)
- **Our Dataset (with attack samples)**: https://drive.google.com/file/d/1230XtlzJn6vdZdvYwc5IiILRJisXmYzn/view?usp=drive_link

### File Structure

#### Blender Dataset Format

```
[dataset]_upload
└── [scene]_13_15_[attack_image]
    ├── attack/
    ├── constraint/
    ├── transform_attack.json
    └── transform_constraint.json
```

#### MipNeRF-360 Dataset Format

```
[dataset]_upload
└── [scene]
    ├── attack/
    ├── constraint/
    ├── transform_attack.json
    └── transform_constraint.json
```

> Please copy the **original dataset content** for each scene into the corresponding directories to construct the complete training data.

After copying the original dataset，the resulting dataset will be structured as shown below:

```
[dataset]_upload
└── [scene]_dir
    ├── attack/
    ├── constraint/
    ├── transform_attack.json
    ├── transform_constraint.json
    └── [all contents in original scene dataset directory]
```

---

## 🏋️‍♂️ Training and Evaluation

### 1. Train a Clean Model

First, train a clean 3DGS model using the original dataset as a baseline.

```bash
# Example command (modify paths accordingly)
python train.py -s path/to/clean_dataset -m path/to/clean_model
```

### 2. Launch the GaussTrap Attack

Use the following command to perform the attack:

```bash
# Start attack training
python train_attack_3epoch.py \
    -m "path/to/model" \
    -s "path/to/attack_dataset" \
    --attack_epochs 2500 \
    --attack_iters 25 \
    --constraint_iters 5 \
    --retrain_iters 25 \
    --densify_until_iter 100000 \
    --start_checkpoint "path/to/clean_model/checkpoint" \
    --save_epoch "save_epoch"
```

### 3. Rendering and Evaluation

After the attack is completed, use the following commands for visualization and performance evaluation:

```bash
# Render attack effect
python render.py --attack True -m "model saving path" --iteration 2500

# Evaluate attack metrics
python metrics_attack.py -m "model saving path" --iteration 2500
```

---

## 📎 Citation

If you find our work useful in your research, please cite:

```bibtex
@article{hong2025gausstrap,
  title={GaussTrap: Stealthy Poisoning Attacks on 3D Gaussian Splatting for Targeted Scene Confusion},
  author={Hong, Jiaxin and Chen, Sixu and Sun, Shuoyang and Yu, Hongyao and Fang, Hao and Tan, Yuqi and Chen, Bin and Qi, Shuhan and Li, Jiawei},
  journal={arXiv preprint arXiv:2504.20829},
  year={2025}
}
```

---

## License

This project includes code from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), which is licensed under a custom non-commercial license by Inria and MPII.

All modifications and additional components created by Jiaxin Hong are licensed under the MIT License.

See [LICENSE](LICENSE) for full details.
