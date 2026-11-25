"""Demo: attack the tensor ring decomposition of one image.

Four perturbations with the same budget are generated (Gaussian noise, ATTR, FAG-AdaTR, AdaTR). Every
perturbed image X + E is decomposed by TR-ALS, and the error of the reconstruction with respect to the
clean image X is printed.

Examples:
    python demo.py --image path/to/image.png --size 128         # quick run, a few minutes on a CPU
    python demo.py --image path/to/image.png --sqrt-eps 500     # main setting of the paper (full size, GPU)
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

import adatr

NAMES = {"gaussian": "Gaussian noise", "attr": "ATTR", "fag_adatr": "FAG-AdaTR", "adatr": "AdaTR"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="path of an RGB image")
    p.add_argument("--size", type=int, default=None, help="resize the image to size x size (default: full resolution)")
    p.add_argument("--rank", type=int, default=5, help="TR-rank R_1 = R_2 = R_3")
    p.add_argument("--sqrt-eps", type=float, default=None, help="budget: bound on ||E||_F (500 in the paper)")
    p.add_argument("--budget", type=float, default=0.05,
                   help="budget as a fraction of ||X||_F, used when --sqrt-eps is not given")
    p.add_argument("--outer", type=int, default=100, help="outer iterations of the attacks")
    p.add_argument("--inner", type=int, default=100, help="inner TR-ALS iterations")
    p.add_argument("--eta", type=float, default=0.1, help="step size of FAG-AdaTR")
    p.add_argument("--lr", type=float, default=0.1, help="Adam learning rate of AdaTR and ATTR")
    p.add_argument("--attacks", nargs="+", default=list(NAMES), choices=list(NAMES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="folder to save X + E and the reconstructions (PNG) and E (.npy)")
    args = p.parse_args()

    image = Image.open(args.image).convert("RGB")
    if args.size:
        image = image.resize((args.size, args.size), Image.LANCZOS)
    X = torch.tensor(np.array(image) / 255.0, dtype=torch.float32, device=args.device)
    sqrt_eps = args.sqrt_eps if args.sqrt_eps is not None else args.budget * torch.norm(X).item()

    torch.manual_seed(args.seed)
    cores_init = adatr.init_cores(X.shape, args.rank, device=args.device)  # known to the attacker and used by TR-ALS
    E_start = adatr.gaussian_noise(X, sqrt_eps)                            # common starting point of the attacks

    print(f"image {tuple(X.shape)}, TR-rank {args.rank}, budget ||E||_F <= {sqrt_eps:.2f} "
          f"({100 * sqrt_eps / torch.norm(X).item():.1f} % of ||X||_F), {args.outer} outer / {args.inner} inner iterations, "
          f"{args.device}")
    print(f"{'attack':<16}{'||E||_F':>9}{'RSE':>9}{'PSNR (dB)':>11}{'time (s)':>10}")

    def evaluate(name, E, seconds=None):
        with torch.no_grad():
            X_hat = adatr.tr_to_tensor(adatr.tr_als(X + E, cores_init, args.inner))
        t = "-" if seconds is None else f"{seconds:.0f}"
        print(f"{name:<16}{torch.norm(E).item():>9.2f}{adatr.rse(X, X_hat):>9.4f}{adatr.psnr(X, X_hat):>11.2f}{t:>10}",
              flush=True)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            tag = name.lower().replace(" ", "_").replace("-", "_")
            save_png(X + E, os.path.join(args.out, f"{tag}_input.png"))
            save_png(X_hat, os.path.join(args.out, f"{tag}_reconstruction.png"))
            np.save(os.path.join(args.out, f"{tag}_E.npy"), E.cpu().numpy())

    evaluate("no attack", torch.zeros_like(X))
    for key in args.attacks:
        start = time.time()
        if key == "gaussian":
            E = E_start
        elif key == "attr":
            E = adatr.attr(X, cores_init, sqrt_eps, args.outer, args.inner, lr=args.lr, E_init=E_start)
        elif key == "fag_adatr":
            E = adatr.fag_adatr(X, cores_init, sqrt_eps, args.outer, args.inner, eta=args.eta, E_init=E_start)
        else:
            E = adatr.adatr(X, cores_init, sqrt_eps, args.outer, args.inner, lr=args.lr, E_init=E_start)
        evaluate(NAMES[key], E, None if key == "gaussian" else time.time() - start)


def save_png(T, path):
    Image.fromarray((T.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(path)


if __name__ == "__main__":
    main()
