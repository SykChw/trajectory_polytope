# Time-Evolving Polytopes from ReLU Networks

This project learns a **time-dependent geometric object** defined implicitly by a function $f(x,y,z,t)$ and reconstructs it as a sequence of **non-convex polytopes evolving over time** using a trained ReLU neural network.

---

# Overview

We model a **family of 3D shapes evolving over time** using a neural network:

- Input: $(x, y, z, t)$
- Output: Binary classification (inside / outside)

The model learns:
$\hat{y} = \mathbf{1}\big[(x,y,z,t) \in \mathcal{P}\big]$

where:
$\mathcal{P}_t = \{(x,y,z) \mid f(x,y,z,t) \ge 0\}$

We then reconstruct $\mathcal{P}_t$ at each time $t$ and visualize its evolution.

---

# Key Idea

Instead of explicitly modeling geometry, we use an **implicit representation**:

$f : \mathbb{R}^3 \times \mathbb{R} \to \mathbb{R}$

At each time $t$, the shape is:

- Interior:
  $f(x,y,z,t) > 0$

- Surface:
  $f(x,y,z,t) = 0$

- Exterior:
  $f(x,y,z,t) < 0$

---

# What the Model Learns

We train a ReLU network:
$\hat{f}(x,y,z,t) \approx \text{sign}(f(x,y,z,t))$

This induces a partition of $\mathbb{R}^4$ into **piecewise-linear regions**.

---

# Mathematical Formalism

---

## 1. Implicit Geometry

A time-dependent shape is defined as:

$\mathcal{P}_t = \{ x \in \mathbb{R}^3 \mid f(x,t) \ge 0 \}$

---

## 2. Neural Approximation

Let:
$g_\theta(x,y,z,t)$

be a ReLU neural network.

We train:
$g_\theta(x,y,z,t) \approx \mathbf{1}[f(x,y,z,t) \ge 0]$

---

## 3. ReLU Geometry

A ReLU network partitions input space into regions:

$\mathbb{R}^4 = \bigcup_{i} R_i$

Within each region:
$g_\theta(x) = W_i x + b_i$

Thus the decision boundary is **piecewise linear**.

---

## 4. Extracted Shape

At time $t$:

$\mathcal{P}_t^{(model)} = \{ (x,y,z) \mid g_\theta(x,y,z,t) \ge 0 \}$

---

## 5. Approximation via Sampling

We approximate:

$\mathcal{P}_t \approx \{ x_i \mid g_\theta(x_i,t) \ge 0 \}$

---

## 6. Alpha Shapes (Non-Convex Geometry)

Given sampled points $S$, we compute:

$\alpha\text{-shape}(S)$

which approximates the underlying geometry.

---

## 7. Volume Tracking

We compute:

$V(t) = \text{Vol}(\mathcal{P}_t)$

to track geometric evolution.

---

# Pipeline

```
f(x,y,z,t)
↓
Dataset Generation
↓
ReLU Network Training
↓
Sampling (x,y,z at fixed t)
↓
Classification (inside/outside)
↓
Alpha Shape Reconstruction
↓
Visualization (GIF)
↓
Volume Tracking
```

---

# Project Structure

```
trajectory_polytope/
│
├── dataset.py
├── dataloader.py
├── model.py
├── train.py
├── load_model.py
├── sampling.py
├── alpha_shape.py
├── volume.py
├── main.py
│
├── frames/
├── outputs/
```

---

# Installation

```bash
pip install torch numpy scipy matplotlib imageio alphashape trimesh shapely
```

---

# Usage

---

## 1. Define your function

In `train.py`:

```python
def f(x, y, z, t):
    cx = torch.sin(2 * torch.pi * t)
    cy = torch.cos(2 * torch.pi * t)
    cz = 0.5 * torch.sin(4 * torch.pi * t)

    r = 0.3 + 0.1 * torch.sin(2 * torch.pi * t)

    return r**2 - ((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
```

---

## 2. Train the model

```bash
python train.py
```

This produces:

```
trajectory_model.pt
```

---

## 3. Extract evolving polytopes

```bash
python main.py
```

---

# Outputs

---

## GIF

```
outputs/evolving.gif
```

Shows the **3D shape evolving over time**

---

## Volume Plot

```
outputs/volume.png
```

Plots:
$V(t)$

---

## Frames

```
frames/frame_000.png
...
```

---

# Key Features

---

## Function-based dataset

You can plug any:

$f(x,y,z,t)$

---

## Non-convex shape extraction

Uses **alpha shapes** instead of convex hulls.

---

## Time evolution

Produces:

${\mathcal{P}*t}*{t}$

---

## Geometry tracking

Tracks:

* shape
* volume
* evolution

---

# Example Variations

---

## Multiple objects

```python
return torch.maximum(f1, f2)
```

---

## Intersection

```python
return torch.minimum(f1, f2)
```

---

## Deforming shapes

```python
return 1 - (x**2/a(t) + y**2/b(t) + z**2/c(t))
```

---

# Conceptual Insight

This project implements:

> **Implicit neural representation of time-dependent geometry**

It connects ideas from:

* Signed Distance Functions (SDFs)
* Neural fields
* ReLU polytope partitioning
* Computational geometry

---

# Limitations

---

## Sampling-based approximation

* Requires many samples
* May miss thin structures

---

## Alpha shape sensitivity

* Depends on parameter (\alpha)

---

## Binary classification

* Only learns sign, not exact distance

---

# Future Work

---

* Use **SDF regression instead of classification**
* Replace alpha shapes with **marching cubes**
* Add **error metrics vs ground truth**
* Extract **explicit polyhedral regions from ReLU network**

---

# Summary

We learn:

$(x,y,z,t) \to \text{inside/outside}$

and reconstruct:

$\mathcal{P}_t$

resulting in a **time-evolving geometric representation learned by a neural network**.