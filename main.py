import os
import numpy as np
import tensorflow as tf
import random

from src.data_utils import create_synthetic_dataset, normalize_data, create_batches
from src.model import create_model
from src.train import train_model
from src.generate import generate_handwriting, plot_stroke

def main():
    print("Starting Handwritten Text Generation Project")

    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # Step 1: Create synthetic dataset
    train_data, val_data = create_synthetic_dataset(num_samples=2000)

    # Step 2: Normalize the data
    train_data_norm, mean, std = normalize_data(train_data)
    val_data_norm, _, _ = normalize_data(val_data)

    # Step 3: Create batches
    batch_size = 32
    train_batches = create_batches(train_data_norm, batch_size)
    val_batches = create_batches(val_data_norm, batch_size)

    print(f"Created {len(train_batches)} training batches and {len(val_batches)} validation batches")

    # Step 4: Create model
    model = create_model()
    model.summary()

    print("\nTraining the model...")
    history = train_model(model, train_batches, val_batches, num_epochs=5)

    # Step 5: Save the model
    model.save("handwriting_model.h5")
    print("Model saved as 'handwriting_model.h5'")

    # Step 6: Generate samples
    print("Generating handwriting samples...")
    os.makedirs("output", exist_ok=True)

    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]
    for temp in temperatures:
        generated_stroke = generate_handwriting(model, mean, std, seq_len=400, temperature=temp)
        plot_stroke(generated_stroke, f"output/sample_temp_{temp}.png")
        print(f"Generated sample with temperature {temp}")

    print("Generated samples saved in 'output' directory")

    # Step 7: Plot training history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("output/training_history.png")
    plt.close()

    print("Training history plot saved as 'output/training_history.png'")
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()
