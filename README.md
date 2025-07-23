# Geometric Inductive Priors in Diffusion-Based Optical Flow Estimation 🌀✨

**🚧 Preliminary Release**  
This repository contains the code for **GA-DDVM**, a Geometric Algebra-based variant of the Denoising Diffusion Vision Model (DDVM), introduced in our paper: **_Geometric Inductive Priors in Diffusion-Based Optical Flow Estimation_**, presented at the Beyond Euclidean Workshop, in conjunction with ICCV 2025 in Honolulu, HI.

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
- Scaling and rotation are learnable, and applied directly to these objects via a second U-Net which sits in GA
- The result: more efficient, interpretable, and targeted convergence to realistic solutions.

---

## 🗂 Repository Structure
