<p align="center">
<img src="https://github.com/Yonsei-HEP-COSMO/DeeLeMa/blob/main/img/DeeLeMa.png?raw=true" width="300">
</p>

# DeeLeMa (Deep Learning for Mass estimation)

We present DeeLeMa, a deep learning network to analyze energies and momenta in particle collisions at high energy colliders, especially DeeLeMa is constructed based on symmetric event topology, and the generated mass distributions show robust peaks at the physical masses after the combinatoric uncertainties, and detector smearing effects are taken into account. DeeLeMa can be widely used in different event topologies by adopting the corresponding kinematic symmetries.

## Requirements

### 1) Pip
To install using pip, simply run the following command:
```
pip3 install -r requirements.txt
```

### 2) PDM (Recommended)

If you don't have `pdm`, you can install it as follows:

```
# Linux / Mac
curl -sSL https://pdm.fming.dev/install-pdm.py | python3 -

# Windows
(Invoke-WebRequest -Uri https://pdm.fming.dev/dev/install-pdm.py -UseBasicParsing).Content | python -
```

Once installed, you can manage dependencies with the following commands:

```
# Install dependencies from pyproject.toml
pdm install

# Activate virtual environment (venv)
source .venv/bin/activate
```

## How to Use
Details on how to use DeeLeMa will be provided soon.

## Citation
If you use DeeLeMa in your research, please cite the following paper:

```
@article{Ban:2022hfk,
    author = "Ban, Kayoung and Kang, Dong Woo and Kim, Tae-Geun and Park, Seong Chan and Park, Yeji",
    title = "{DeeLeMa: Missing Information Search with Deep Learning for Mass Estimation}",
    eprint = "2212.12836",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "12",
    year = "2022"
}
```

## Reference

* K. Ban, D. W. Kang, T.-G Kim, S. C. Park and Y. Park,  *DeeLeMa: Missing Information Search with Deep Learning for Mass Estimation*, [arXiv:2212.12836](https://arxiv.org/abs/2212.12836)
