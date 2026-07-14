# CompCars Dataset Statistics

generated: 2026-02-28 19:35:42

## Source

**URL:** http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html

**Citation:**
> Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. "A Large-Scale Car Dataset for Fine-Grained Categorization and Verification", In Computer Vision and Pattern Recognition (CVPR), 2015.

## Overview

| Split | Images | Makes | Models | Years |
|-------|--------|-------|--------|-------|
| Train | 16016 | 75 | 431 | 13 |
| Test | 14939 | 75 | 431 | 13 |
| Total | 30955 | - | - | - |

## Make Classification (Coarse-grained)

| Metric | Train | Test |
|--------|-------|------|
| Number of classes | 75 | 75 |
| Min samples/class | 10 | 9 |
| Max samples/class | 1201 | 1125 |
| Mean samples/class | 213.5 | 199.2 |
| Std samples/class | 228.7 | 215.5 |

## Model Classification (Fine-grained)

| Metric | Train | Test |
|--------|-------|------|
| Number of classes | 431 | 431 |
| Min samples/class | 10 | 8 |
| Max samples/class | 143 | 140 |
| Mean samples/class | 37.2 | 34.7 |
| Std samples/class | 21.4 | 21.3 |

## Class Imbalance Analysis

- make imbalance ratio (max/min): 120.1x
- model imbalance ratio (max/min): 14.3x

## Figures

- `make_distribution.png` - samples per make class
- `model_distribution.png` - samples per model class
- `year_distribution.png` - images per year
