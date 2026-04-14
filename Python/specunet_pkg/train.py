import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

from .utils import get_model_path_filename
from .logger import log_print

def save_loss_graphs(model_name, training_loss_epochs, testing_loss_epochs, num_epochs, logger):
    x_labels = np.arange(1, num_epochs)
    training_losses = training_loss_epochs[1:]  # Ignore 1st epoch
    testing_losses = testing_loss_epochs[1:]
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, training_losses, label="Training RMSE", color="b", marker="o")
    plt.plot(x_labels, testing_losses, label="Testing Data", color="r", marker="o")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.legend()

    if model_name == 'unet':
        model_name = "conv_UNet"

    plt.savefig(f"{model_name}_training_testing_rmse_graph.png")

    np.save(f'{model_name}_training_losses.npy', np.array(training_losses))
    np.save(f'{model_name}_testing_losses.npy', np.array(testing_losses))

    log_print(logger, f"[train] {model_name} training and testing RMSE graph saved!")


def save_model(model_name, model, input_size, save_model_onnx, logger):
    """Saves model weights and exports to ONNX."""
    model_path = get_model_path_filename(model_name, "base")
    torch.save(model.state_dict(), model_path)

    model.eval()
    dummy_input = torch.randn(input_size, dtype=torch.float64, device='cuda').unsqueeze(1)
    traced_model = torch.jit.trace(model, dummy_input)

    log_print(logger, f"[train] Model saved at: {os.path.abspath(model_path)}")

    # Export to ONNX
    if save_model_onnx:
        onnx_path = get_model_path_filename(model_name, "onnx")
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          export_params=True,
                          opset_version=18,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
        log_print(logger, f"[train] Model saved as .onnx format at {os.path.abspath(onnx_path)}")


def train(model_name, model, train_loader, val_loader, criterion, optimizer, scheduler, logger, device, opt):

    training_loss_epochs = []
    val_loss_epochs = []

    opt_model = opt["model"]
    num_epochs = opt_model["hyperparameters"]["epochs"]

    try:
        for epoch in range(num_epochs):
            model.train()
            training_loss = 0.0

            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                training_loss += loss.item() * inputs.size(0)

            training_loss_epoch = training_loss / len(train_loader.dataset)
            log_print(logger, f"[train] Epoch [{epoch + 1}/{num_epochs}], Training Loss: {training_loss_epoch}")

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            val_loss_epoch = val_loss / len(val_loader.dataset)
            log_print(logger, f"[train] Epoch [{epoch + 1}/{num_epochs}], Testing Loss: {val_loss_epoch}")

            model.train()
            scheduler.step()

            training_loss_epochs.append(training_loss_epoch)
            val_loss_epochs.append(val_loss_epoch)

    except KeyboardInterrupt as i:
        log_print(logger, "\n[!] Training manually interrupted by user. Cancelling training.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error occured during training: {e}")

    save_loss_graphs(model_name, training_loss_epochs, val_loss_epochs, num_epochs, logger)

    save_model_onnx = opt['exp_path']['save_model_onnx']
    save_model(model_name, model, opt_model["input_size"], save_model_onnx, logger)

    log_print(logger, "[train] Training complete!")

    return model

# def train(model_name, model, train_loader, test_loader, criterion, optimizer, scheduler, logger, device, args):
#     class MetricTracker:
#         """
#         Records training or validation numerical indicators over time.
#         """
#
#         def __init__(self, *keys, phase='train'):
#             self.phase = phase
#             self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
#             self.reset()
#
#         def reset(self):
#             for col in self._data.columns:
#                 self._data[col].values[:] = 0
#
#         def update(self, key, value, n=1):
#             self._data.total[key] += value * n
#             self._data.counts[key] += n
#             self._data.average[key] = self._data.total[key] / self._data.counts[key]
#
#         def avg(self, key):
#             return self._data.average[key]
#
#         def result(self):
#             return {'{}/{}'.format(self.phase, k): v for k, v in dict(self._data.average).items()}
#
#     training_loss_epochs = []
#     testing_loss_epochs = []
#
#     train_metrics = MetricTracker('loss', phase='train')
#     val_metrics = MetricTracker('loss', phase='val')
#
#     for epoch in range(args.epochs):
#         model.train()
#         train_metrics.reset()
#
#         for inputs, targets, _ in train_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#
#             train_metrics.update('loss', loss.item(), n=inputs.size(0))
#
#         train_result = train_metrics.result()
#         training_loss_epoch = train_result['train/loss']
#         log_print(logger, f"Epoch [{epoch + 1}/{args.epochs}], Training Loss: {training_loss_epoch:.6f}")
#
#         model.eval()
#         val_metrics.reset()
#
#         with torch.no_grad():
#             for inputs, targets, _ in test_loader:
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#
#                 val_metrics.update('loss', loss.item(), n=inputs.size(0))
#
#         val_result = val_metrics.result()
#         testing_loss_epoch = val_result['val/loss']
#         log_print(logger, f"Epoch [{epoch + 1}/{args.epochs}], Testing Loss: {testing_loss_epoch:.6f}")
#
#         model.train()
#         scheduler.step()
#
#         training_loss_epochs.append(training_loss_epoch)
#         testing_loss_epochs.append(testing_loss_epoch)
#
#     save_loss_graphs(model_name, training_loss_epochs, testing_loss_epochs, args.epochs, logger)
#     save_model(model_name, model, args.input_size, logger)
#
#     log_print(logger, "Training complete!")
#
#     return model

# def train(model_name, model, train_loader, test_loader, criterion, optimizer, scheduler, logger, device, args):
#
#     training_loss_epochs = []
#     testing_loss_epochs = []
#
#     # Training loop
#     for epoch in range(args.epochs):
#         model.train()
#         training_loss = 0.0
#
#         for inputs, targets, _ in train_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#
#             training_loss += loss.item() * inputs.size(0)
#
#         training_loss_epoch = training_loss / len(train_loader.dataset)
#
#         log_print(logger, f"Epoch [{epoch + 1}/{args.epochs}], Training Loss: {training_loss_epoch}")
#
#         model.eval()
#         testing_loss = 0.0
#
#         with torch.no_grad():
#             for inputs, targets, _ in test_loader:
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 testing_loss += loss.item() * inputs.size(0)
#
#         testing_loss_epoch = testing_loss / len(test_loader.dataset)
#         log_print(logger,f"Epoch [{epoch + 1}/{args.epochs}], Testing Loss: {testing_loss_epoch}")
#
#         model.train()
#         scheduler.step()
#
#         training_loss_epochs.append(training_loss_epoch)
#         testing_loss_epochs.append(testing_loss_epoch)
#
#     save_loss_graphs(model_name, training_loss_epochs, testing_loss_epochs, args.epochs, logger)
#
#     save_model(model_name, model, args.input_size, logger)
#
#     log_print(logger, "Training complete!")
#
#     return model