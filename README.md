# AdaTR: Adversarial Attack on Tensor Ring Decomposition

Code for the paper *Adversarial Attack on Tensor Ring Decomposition* (Peilin Yang, Yuning Qiu,
Toshihisa Tanaka, Guoxu Zhou, Qibin Zhao).

## What the paper does

Tensor ring (TR) decomposition approximates a tensor by a ring of small third-order cores. It is used
for image and video recovery, for the compression of neural networks, and in recommender systems.
The paper asks how much a small perturbation of the data can damage this decomposition.

An attacker adds a perturbation `E` with `||E||_F <= sqrt_eps` to a tensor `X`. The user only receives
`X + E` and decomposes it by TR-ALS. The damage is the error of the resulting low-rank approximation
with respect to the clean tensor:

```
maximize over E     || X - TR(G(E)) ||_F^2
where               G(E) = cores returned by TR-ALS on X + E
subject to          || E ||_F^2 <= eps
```

This is a bilevel problem: the cores are fitted to the perturbed tensor, and the error is measured on
the clean one. The repository contains the two proposed attacks and the baseline they are compared with.

| Function | Method | How the perturbation is updated |
|---|---|---|
| `adatr.adatr` | **AdaTR** (Algorithm 1) | Gradient of the objective above, by back-propagation through all inner TR-ALS iterations |
| `adatr.fag_adatr` | **FAG-AdaTR** (Algorithm 2) | Closed-form approximate gradient `E_[n] P_n`, without back-propagation; much less memory |
| `adatr.attr` | ATTR baseline | Ascent on the fitting error `\|\|X + E - TR(G)\|\|_F^2` (extension of adversarially-trained NMF to TR decomposition) |
| `adatr.gaussian_noise` | Gaussian noise | Random noise scaled to the budget |

## Files

```
adatr/tr_als.py    TR-ALS: initial cores, core update, decomposition, reconstruction
adatr/attacks.py   adatr(), fag_adatr(), attr(), gaussian_noise()
adatr/metrics.py   RSE and PSNR
demo.py            runs the four perturbations on one image and prints the results
```

## Installation

Python 3.9 or later with PyTorch, NumPy, and Pillow:

```
git clone https://github.com/justicbro/adversarial-attack-on-tensor-ring.git
cd adversarial-attack-on-tensor-ring
pip install -r requirements.txt
```

The code runs on a CPU for small tensors and on a GPU for full-size images.

## Quick start

```
python demo.py --image path/to/image.png --size 128
```

This resizes the image to 128 x 128 x 3, generates the four perturbations with the same budget
(5 % of `||X||_F` unless `--sqrt-eps` or `--budget` is given), decomposes every perturbed image by
TR-ALS with TR-rank 5, and prints the error of the reconstruction with respect to the clean image.
On a CPU it takes a few minutes. Example output for one DIV2K image:

```
image (128, 128, 3), TR-rank 5, budget ||E||_F <= 7.03 (5.0 % of ||X||_F), 100 outer / 100 inner iterations, cpu
attack            ||E||_F      RSE  PSNR (dB)  time (s)
no attack            0.00   0.0436      31.18         -
Gaussian noise       7.03   0.0459      30.71         -
ATTR                 7.03   0.0468      30.55       114
FAG-AdaTR            7.03   0.0655      27.63       120
AdaTR                7.03   0.0777      26.16       141
```

All perturbations have the same norm. Gaussian noise and ATTR hardly change the error of TR-ALS, while
FAG-AdaTR and AdaTR increase it clearly. Add `--out results/` to save the perturbed images, the
reconstructions (PNG), and the perturbations (`.npy`). `python demo.py -h` lists all options.

## Using the functions

```python
import torch
import adatr

X = ...                                   # clean tensor, e.g. an image of size H x W x 3 with values in [0, 1]
sqrt_eps = 0.05 * torch.norm(X).item()    # budget: ||E||_F <= sqrt_eps

torch.manual_seed(0)
cores_init = adatr.init_cores(X.shape, ranks=5, device=X.device)   # initial cores of TR-ALS

E = adatr.adatr(X, cores_init, sqrt_eps)        # or adatr.fag_adatr(...), adatr.attr(...)

cores = adatr.tr_als(X + E, cores_init)         # the user's decomposition of the perturbed tensor
X_hat = adatr.tr_to_tensor(cores)
print(adatr.rse(X, X_hat), adatr.psnr(X, X_hat))
```

The functions accept tensors of any order of at least three (for example videos of size
H x W x 3 x frames). `ranks` is one integer or the list `(R_1, ..., R_N, R_1)`.

## Settings of the paper

The main experiments use TR-rank 5, the budget `sqrt_eps = 500` on full-size images, 100 outer and
100 inner iterations, the step size `eta = 0.1` for FAG-AdaTR and the Adam learning rate `0.1` for
AdaTR and ATTR, and single precision. Except for the budget, these are the defaults of the functions
and of `demo.py`:

```
python demo.py --image path/to/image.png --sqrt-eps 500
```

The images are from the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset and the videos from
the [YUV sequences](http://trace.eas.asu.edu/yuv/). They are not included in this repository.

Notes:

- The attacker generates the perturbation with the TR-ranks and the initial cores of TR-ALS. In the
  demo, the same initial cores are used to decompose the perturbed image.
- AdaTR stores the computational graph of all inner iterations. A full-size image with TR-rank 5 and
  100 inner iterations fits on a GPU with 24 GB of memory. FAG-AdaTR needs far less memory.
- TR-ALS stops before the given number of iterations if the relative error does not change between
  two iterations (`tol` in `tr_als`).

## Citation

```
@article{yang2026adatr,
  title   = {Adversarial Attack on Tensor Ring Decomposition},
  author  = {Yang, Peilin and Qiu, Yuning and Tanaka, Toshihisa and Zhou, Guoxu and Zhao, Qibin},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
