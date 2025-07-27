# Geometric Inductive Priors in Diffusion-Based Optical Flow Estimation 🌀✨

**🚧 Preliminary Release**  
This repository contains the code for **GA-DDVM**, a Geometric Algebra-based variant of the Denoising Diffusion Vision Model (DDVM), introduced in our paper:  
**_Geometric Inductive Priors in Diffusion-Based Optical Flow Estimation_**, presented at the Beyond Euclidean Workshop, in conjunction with ICCV 2025 in Honolulu, HI.

---

## 🧠 Overview

GA-DDVM incorporates geometric inductive biases into the DDVM pipeline for optical flow estimation through a second, cascaded U-Net which is smaller yet more specialized.  
It constrains the generative process in two key ways:

1. 🌀 **Object Constraint**: The model learns only 2D vector fields (i.e., optical flows).
2. 🔄 **Operation Constraint**: Network layers are restricted to scaling and rotations, operations that preserve the geometry of flow fields.

These constraints accelerate convergence and improve accuracy.

---

## 📊 Key Results

At 600k training steps, GA-DDVM outperforms DDVM baseline models significantly given the same training setup:

- **KITTI**
  - 🟢 Endpoint Error (EPE): ↓ 76.3%
  - 🟢 Fl-all: ↓ from 76.8% → 20.1%
- **Sintel**
  - 🟢 Clean: ↓ from 11.4 → 3.38
  - 🟢 Final: ↓ from 11.7 → 4.46

These gains appear early and persist across training and testing alike.

---

## 📐 Why Geometric Algebra?

Geometric Algebra (GA) provides a natural way to represent and manipulate geometric objects like vectors and transformations.  
In GA-DDVM:

- The efficient U-Net backbone extracts features, which are reshaped as proposals of 2D vector fields (optical flow),
- Optical flow is modeled as a 2D vector within G(2,0,0), the 2D Geometric Algebra.
- Scaling and rotation are learnable, and applied directly to these objects via a second U-Net which sits in GA.
- The result: more efficient, interpretable, and targeted convergence to realistic solutions.

---

## 🏃‍♂️ Training and Execution

### 1. Set Up Open-DDVM on Linux

1. Clone the repository:

    ```bash
    git clone https://github.com/DQiaole/FlowDiffusion_pytorch.git
    ```

2. Create and activate an empty environment:

    ```bash
    micromamba create --name ddvm
    micromamba activate ddvm
    ```

3. Install Python (recommended version 3.8.0):

    ```bash
    micromamba install python==3.8.0
    ```

4. Edit and complete `requirements.txt` as follows to avoid conflicts:

    ```txt
    azure-ai-ml
    azure-core
    azure-identity
    azure-mgmt-authorization
    jsonargparse
    pip-requirements-parser

    # DDVM requirements
    imagen-pytorch
    tensorflow-cpu==2.13.0
    tensorflow-addons==0.21.0
    diffusers[torch]==0.20.2
    opencv-python-headless
    ml-collections
    scipy
    accelerate==0.21.0
    ```

5. Install the required dependencies:

    ```bash
    python -m pip install -r requirements.txt
    ```

6. Obtain the necessary `datasets` and, if required, `checkpoints`.

---

### 2. Running Training Scripts

#### Imagen_CliffordUnet

```bash
python train_flow_prediction.py --stage 'autoflow' --train_batch_size 1 --image_size 320 448 --dataloader_num_workers 16 --num_steps 1000000 --save_images_steps 500 --gradient_accumulation_steps 1 --lr_warmup_steps 10000 --use_ema --mixed_precision 'bf16' --prediction_type 'sample' --ddpm_num_steps 64 --checkpointing_steps 10000 --checkpoints_total_limit 5 --output_dir "check_points/autoflow-CliffordImagenUnet" --Unet_type 'CliffordSRUnet256' --max_flow 400 --learning_rate 1e-4 --adam_weight_decay 0.0001 --it_aug --add_gaussian_noise --normalize_range --lr_scheduler 'cosine'
```

#### RAFT_CliffordUnet

```bash
python train_flow_prediction.py --stage 'autoflow' --train_batch_size 1 --image_size 320 448 --dataloader_num_workers 16 --num_steps 1000000 --save_images_steps 500 --gradient_accumulation_steps 1 --lr_warmup_steps 10000 --use_ema --mixed_precision 'bf16' --prediction_type 'sample' --ddpm_num_steps 64 --checkpointing_steps 10000 --checkpoints_total_limit 5 --output_dir "check_points/autoflow-RAFT_CliffordUnet" --Unet_type 'RAFT_CliffordUnet' --max_flow 400 --learning_rate 1e-4 --adam_weight_decay 0.0001 --it_aug --add_gaussian_noise --normalize_range --lr_scheduler 'cosine'
```

---

### 3. Training with Accelerate

To train using multiple processes (e.g., GPUs):

```bash
accelerate launch --num_processes 8 train_flow_prediction.py --stage 'autoflow' --train_batch_size 4 --image_size 320 448 --dataloader_num_workers 16 --num_steps 1000000 --save_images_steps 500 --gradient_accumulation_steps 1 --lr_warmup_steps 10000 --use_ema --mixed_precision 'bf16' --prediction_type 'sample' --ddpm_num_steps 64 --checkpointing_steps 10000 --checkpoints_total_limit 5 --output_dir "check_points/autoflow-ImagenUnet" --max_flow 400 --learning_rate 1e-4 --adam_weight_decay 0.0001 --it_aug --add_gaussian_noise --normalize_range --lr_scheduler 'cosine'
```


---

### 4. Launch Jobs on AzureML

To run jobs in AzureML:

```bash
PythonCommandPrompt.cmd
activate venv
pip install -r requirements.txt
azcopy login
azcopy copy --recursive "source_dir" "azure_dir"
python submit_aml_job.py --config testing_config.yaml
```

---

### 5. Evaluation

Evaluate a trained pipeline (e.g., after 305k steps):

```bash
python evaluate_diffusers_warprefine.py --pipeline_path check_points/autoflow-CorrUnet/pipeline-305000 --normalize_range --validation kitti sintel
```

Sample results:

```
Validation KITTI: EPE: 5.702541 Fl-all: 19.236995
VALIDATION Sintel: EPE {'final': 3.9618032, 'clean': 3.0904233}
```

## 📚 How to Cite

If you use this repository or ideas from our work, please consider citing the following:

```bibtex
@inproceedings{pepe2025geometric,
  title={Geometric Inductive Priors in Diffusion-Based Optical Flow Estimation},
  author={Pepe, Alberto and Mendonca, Paulo RS and Lasenby, Joan},
  booktitle={2nd Beyond Euclidean Workshop: Hyperbolic and Hyperspherical Learning for Computer Vision},
  year={2025}
}

@article{saxena2023surprising,
  title={The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular Depth Estimation},
  author={Saxena, Saurabh and Herrmann, Charles and Hur, Junhwa and Kar, Abhishek and Norouzi, Mohammad and Sun, Deqing and Fleet, David J},
  journal={arXiv preprint arXiv:2306.01923},
  year={2023}
}

@misc{dong2023openddvm,
  title = {Open-DDVM: A Reproduction and Extension of Diffusion Model for Optical Flow Estimation},
  author = {Dong, Qiaole and Zhao, Bo and Fu, Yanwei},
  journal = {arXiv preprint arXiv:2312.01746},
  year = {2023}
}
```
---
