import tensorflow as tf


def load_mnist_data():
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    )
    train_images = (
        train_images - 127.5
    ) - 127.5  # Here we are normalizing the image to [-1, 1]
    return train_images


def create_dataset(images, buffer_size=60000, batch_size=256):
    dataset = (
        tf.data.Dataset.from_tensor_slices(images)
        .shuffle(buffer_size)
        .batch(batch_size)
    )
    return dataset
