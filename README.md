# InfoGAN in PyTorch

Another InfoGAN implementation in PyTorch

Tested with `torch==1.9` and Python `3.7`.

## Usage

### Dependencies
```
pip install torch
```

### Training
```
python3 main.py --root-path .
```

### Plotting
```
python3 plot.py --root-path .
```

## Results

### Categorical latent code:

![mnist](http://mhocke.userpage.fu-berlin.de/courses/igml/figures/mnist.png)

### First continuous latent code (rotation):

![rotation](http://mhocke.userpage.fu-berlin.de/courses/igml/figures/rotation.png)

### Second continuous latent code (width?):

![width](http://mhocke.userpage.fu-berlin.de/courses/igml/figures/width.png)

## Original paper

Chen, Xi, et al. "Infogan: Interpretable representation learning by information
maximizing generative adversarial nets." [arXiv preprint arXiv:1606.03657](https://arxiv.org/pdf/1606.03657.pdf) (2016).
