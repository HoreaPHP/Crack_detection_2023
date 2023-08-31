import matplotlib.pyplot as plt
import os

def plot_line(epochs, what, name, file):
    plt.figure()
    plt.plot(epochs, what, 'r', label=name)
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    # plt.ylim([0, 1])
    plt.legend()
    plt.savefig(f"epoch{len(epochs)}_{file}")
    plt.close()
    plt.show()


def plot_loss_n_acc(epochs, loss, acc, val_loss, val_acc):
    plot_line(epochs, loss, "Training Loss", "loss")
    plot_line(epochs, val_loss, "Validation Loss", "val_loss")
    plot_line(epochs, acc, "Training Accuracy", "acc")
    plot_line(epochs, val_acc, "Validation Accuracy", "val_acc")


def plot_history(model_history):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    epochs = range(len(loss))
    plot_loss_n_acc(epochs, loss, acc, val_loss, val_acc)