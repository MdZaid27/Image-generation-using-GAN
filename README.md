# MNIST GAN (Generative Adversarial Network)

This repository contains an implementation of a Generative Adversarial Network (GAN) for generating handwritten digit images using the MNIST dataset. The GAN consists of two neural networks, a generator and a discriminator, that are trained in an adversarial manner.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/mnist-gan.git
```

2. Navigate to the project directory:

```
cd mnist-gan
```

3. Install the required dependencies:

```
pip install tensorflow numpy
```

## Usage

1. Run the `train.py` script to train the GAN:

```
python train.py
```

This script will load the MNIST dataset, build the generator and discriminator models, define the training step and loop, and train the GAN for a specified number of epochs (default: 50). During training, it will generate and save images from the generator every epoch.

2. After training is complete, you can find the generated images in the `images` directory.

## Project Structure

- `data/`: Contains the code for loading and preprocessing the MNIST dataset.
- `models/`: Contains the code for building the generator and discriminator models.
- `utils/`: Contains utility functions for loss calculations and image visualization.
- `train.py`: The main script for training the GAN.

## Customization

You can customize various aspects of the GAN by modifying the hyperparameters in the `train.py` script:

- `BUFFER_SIZE`: The buffer size for shuffling the dataset.
- `BATCH_SIZE`: The batch size for training.
- `EPOCHS`: The number of epochs for training.
- `latent_dim`: The dimension of the latent space for the generator.
- `num_examples_to_generate`: The number of examples to generate during training.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
