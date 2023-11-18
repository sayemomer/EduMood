# utils.py

import torch
import matplotlib.pyplot as plt

def save_model(model, save_path):
    """Saves the model to the specified path."""
    torch.save(model.state_dict(), save_path)

def load_model(model, load_path):
    """Loads a model's state from a specified path."""
    model.load_state_dict(torch.load(load_path))
    return model

def load_model_for_testing(model, load_path):
    """Loads a model's state from a specified path."""
    state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    new_state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model


def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plots training loss and accuracy over epochs."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot for Loss
    axs[0].plot(train_losses, label='Training Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_title('Loss Over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot for Accuracy
    axs[1].plot(train_accuracies, label='Training Accuracy')
    axs[1].plot(val_accuracies, label='Validation Accuracy')
    axs[1].set_title('Accuracy Over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.show()