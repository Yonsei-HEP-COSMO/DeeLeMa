<p align="center">
<img src="https://github.com/Yonsei-HEP-COSMO/DeeLeMa/blob/main/img/DeeLeMa.png?raw=true" width="300">
</p>

# DeeLeMa (Deep Learning for Mass estimation)

## Overview

DeeLeMa is a deep learning network designed to analyze energies and momenta in particle collisions at high-energy colliders. Built with a foundation on symmetric event topology, DeeLeMa's generated mass distributions demonstrate robust peaks at the physical masses, even after accounting for combinatoric uncertainties and detector smearing effects. With its adaptability to different event topologies, DeeLeMa's effectiveness shines when corresponding kinematic symmetries are adopted.
<p align="center">
<img src="https://github.com/Yonsei-HEP-COSMO/DeeLeMa/blob/main/img/topology.png?raw=true" width="300">
</p>
## Requirements

### Using Pip

```bash
pip3 install -r requirements.txt
```

### Using PDM (Recommended)

If you haven't installed `pdm` yet:

```bash
# Linux / Mac
curl -sSL https://pdm.fming.dev/install-pdm.py | python3 -

# Windows
(Invoke-WebRequest -Uri https://pdm.fming.dev/dev/install-pdm.py -UseBasicParsing).Content | python -
```

With PDM installed:

```bash
# Install dependencies from pyproject.toml
pdm install

# Activate virtual environment (venv)
source .venv/bin/activate
```

## Getting Started

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Yonsei-HEP-COSMO/DeeLeMa.git
    ```

2. **Install Dependencies**: 
   
   Follow the [Requirements](#requirements) section for instructions.

3. **Training**:

    ⚠️ **Caution**
    > Before training, ensure you modify the data path in `train.py` to point to the location of your data.
    > For more details, refer to [`train.py`](./train.py).

   To train the model, execute the following command:

    ```bash
    python train.py
    ```

4. **Monitoring**:
   
   To monitor the training process, run `tensorboard`:

    ```bash
    tensorboard --logdir=logs/
    ```

    ⚠️ **Caution**
    > If you use PDM then should run tensorboard in activated virtual environment.
  
    

## Citation

If DeeLeMa benefits your research, please acknowledge our efforts by citing the following paper:

```bibtex
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

* K. Ban, D. W. Kang, T.-G Kim, S. C. Park, and Y. Park,  *DeeLeMa: Missing Information Search with Deep Learning for Mass Estimation*, [arXiv:2212.12836](https://arxiv.org/abs/2212.12836)

## License

DeeLeMa is released under the MIT License. For more details, see the `LICENSE` file in the repository.
