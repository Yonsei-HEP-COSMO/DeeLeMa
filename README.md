<p align="center">
<img src="https://github.com/Yonsei-HEP-COSMO/DeeLeMa/blob/main/img/DeeLeMa.png?raw=true" width="300">
</p>

# DeeLeMa (Deep Learning for Mass estimation)

## Overview

$\textsf{DeeLeMa}$ is a deep learning network designed to analyze energies and momenta in particle collisions at high-energy colliders. Built with a foundation on symmetric event topology, $\textsf{DeeLeMa}$'s generated mass distributions demonstrate robust peaks at the physical masses, even after accounting for combinatoric uncertainties and detector smearing effects. With its adaptability to different event topologies, $\textsf{DeeLeMa}$'s effectiveness shines when corresponding kinematic symmetries are adopted.

The current version of $\textsf{DeeLeMa}$ (v1.0.0) is constructed on the $t\bar{t}$-like antler event topology which is shown below figure.
<p align="center">
<img src="https://github.com/Yonsei-HEP-COSMO/DeeLeMa/blob/main/img/topology.png?raw=true" width="350">
    <br>
    <m>$t\bar{t}$-like antler event topology</m>
</p>

## Requirements

### Using Pip

```bash
pip3 install -r requirements.txt
```

### Using Huak (Recommended)

If you haven't installed `huak` yet:

```bash
pip3 install huak
```

With Huak installed:

```bash
# Install dependencies from pyproject.toml
huak install

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
    > If you use huak then should run tensorboard in activated virtual environment.

5. **Testing**:

    - Load the saved checkpoint using the `load_from_checkpoint()` method:

    ```python
    checkpoint_path = "DeeLeMa_Toy.ckpt"
    model = DeeLeMa.load_from_checkpoint(checkpoint_path)
    ```

    - Set the model to evaluation mode:

    ```python
    model.eval()
    ```

    - Use the loaded model for inference or further analysis:

    ```python
    from deelema.utils import decode_missing_momentum

    output = decode_missing_momentum(model, dl_test, m_C) # m_C is the pre-determined mass
    ```

## Citation

If $\textsf{DeeLeMa}$ benefits your research, please acknowledge our efforts by citing the following paper:

```bibtex
@article{Ban:2023mjy,
    author = "Ban, Kayoung and Kang, Dong Woo and Kim, Tae-Geun and Park, Seong Chan and Park, Yeji",
    title = "{Missing information search with deep learning for mass estimation}",
    doi = "10.1103/PhysRevResearch.5.043186",
    journal = "Phys. Rev. Res.",
    volume = "5",
    number = "4",
    pages = "043186",
    year = "2023"
}
```

## Reference

* K. Ban, D. W. Kang, T.-G. Kim, S. C. Park, and Y. Park,  *Missing Information Search with Deep Learning for Mass Estimation*, [PhysRevResearch.5.043186](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043186)

## License

$\textsf{DeeLeMa}$ is released under the MIT License. For more details, see the `LICENSE` file in the repository.



5. **Loading a Trained Model**:

    - Load the saved checkpoint using the `load_from_checkpoint()` method:

    ```python
    checkpoint_path = "DeeLeMa_Toy.ckpt"
    model = DeeLeMa.load_from_checkpoint(checkpoint_path)
    ```

    - Set the model to evaluation mode:

    ```python
    model.eval()
    ```

    - Use the loaded model for inference or further analysis:

    ```python
    predictions = []
    with torch.no_grad():
        for batch in dl_test:
            outputs = model(batch)
            predictions.append(outputs)
    ```

## Citation

If $\textsf{DeeLeMa}$ benefits your research, please acknowledge our efforts by citing the following paper:

```bibtex
@article{Ban:2023mjy,
  author = "Ban, Kayoung and Kang, Dong Woo and Kim, Tae-Geun and Park, Seong Chan and Park, Yeji",
  title = "{Missing information search with deep learning for mass estimation}",
  doi = "10.1103/PhysRevResearch.5.043186",
  journal = "Phys. Rev. Res.",
  volume = "5",
  number = "4",
  pages = "043186",
  year = "2023"
}
```

## Reference

_K. Ban, D. W. Kang, T.-G. Kim, S. C. Park, and Y. Park,_ Missing Information Search with Deep Learning for Mass Estimation*, [PhysRevResearch.5.043186](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043186)

## License

$\textsf{DeeLeMa}$ is released under the MIT License. For more details, see the `LICENSE` file in the repository.
