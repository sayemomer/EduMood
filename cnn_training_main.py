import torch
from torch import optim, nn

from Model.model import MultiLayerFCNet
from DataLoder.dataLoader import custom_loader
from TrainUtils.train_utils import train_epoch, validate_epoch
import Config.config as config
from Utils.utils import save_model, plot_training_results

if __name__ == '__main__':

    train_loader, val_loader = custom_loader(config.BATCH_SIZE, config.TRAIN_DATA_PATH)

    model = MultiLayerFCNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    model = nn.DataParallel(model)
    model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config.LEARNING_RATE)
    best_acc = 0

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    print(config.EPOCHS)

    for epoch in range(config.EPOCHS):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, config.DEVICE)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Save the best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            save_model(model, config.MODEL_SAVE_PATH)

        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Train Loss: {train_loss}, Train Acc: {train_accuracy * 100}%, Val Loss: {val_loss}, Val Acc: {val_accuracy * 100}%")

    # Plotting using utility function
    plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies)
