# SpringDiffusion
An simple example of a diffusion model to generate flowers, written in python.

- [Getting started](#getting-started)
  * [Install](#install)
  * [Usage](#usage)

## Getting started
### Install
Clone this repo and install requirements:

```
git clone https://github.com/killian31/SpringDiffusion.git
cd SpringDiffusion
pip install requirements.txt --upgrade
``` 
### Usage
Run this in your terminal:

```
python3 train_model.py --img-size 256 --batch-size 8 --sampling-steps 1000 --epochs 20 --lr 0.001
```
