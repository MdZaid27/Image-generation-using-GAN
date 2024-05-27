import os
import matplotlib.pyplot as plt
import tensorflow as tf


def generate_and_save_images(model, epoch, test_input, save_dir="images"):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    # Debug prints to trace execution
    print(f"Checking if directory exists: {save_dir}")
    if not os.path.exists(save_dir):
        print(f"Directory does not exist. Creating directory: {save_dir}")
        os.makedirs(save_dir)
    else:
        print(f"Directory exists: {save_dir}")

    save_path = os.path.join(save_dir, f"image_at_epoch_{epoch:04d}.png")
    print(f"Saving image to: {save_path}")
    plt.savefig(save_path)
    plt.close(fig)
