import os
import torch
import numpy as np
import torch.nn as nn
import logging

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 10, verbose: bool = False, delta: float = 0,
                 save_model_folder: str = "", save_model_name: str = "model",
                 logger: logging.Logger = None, model_name: str = None):
        """
        :param patience: int, How long to wait after last time validation loss improved. Default: 10
        :param verbose: bool, If True, prints a message for each validation loss improvement. Default: False
        :param delta: float, Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        :param save_model_folder: str, folder to save the model
        :param save_model_name: str, name of the saved model
        :param logger: logger
        :param model_name: str, name of the model
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf
        self.delta = delta

        self.save_model_folder = save_model_folder
        self.save_model_name = save_model_name
        
        self.logger = logger
        self.model_name = model_name

    def step(self, val_metric_indicator, model):
        """
        update the early stopping state
        :param val_metric_indicator: list of tuple, contains the metric name, value and whether higher is better
        :param model: nn.Module, the model
        :return:
        """
        # score is the first metric in the val_metric_indicator list
        metric_name, score, higher_is_better = val_metric_indicator[0]

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif (score < self.best_score + self.delta and not higher_is_better) or \
             (score > self.best_score + self.delta and higher_is_better):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module, is_best_test: bool = False):
        """
        Saves model when validation metric improves.
        :param model: nn.Module, the model
        :param is_best_test: bool, if True, save a separate checkpoint for the best test model
        """
        # Save the model that performs best on the validation set
        save_path = os.path.join(self.save_model_folder, f"{self.save_model_name}.pkl")
        torch.save(model.state_dict(), save_path)
        
        if is_best_test:
            best_test_save_path = os.path.join(self.save_model_folder, f"{self.save_model_name}_best_test.pkl")
            self.logger.info(f"Saving current model as best on test set to {best_test_save_path}")
            torch.save(model.state_dict(), best_test_save_path)

    def load_checkpoint(self, model: nn.Module, is_best_test: bool = False):
        """
        Loads model checkpoint.
        :param model: nn.Module, the model
        :param is_best_test: bool, if True, load the checkpoint of the best test model
        """
        if is_best_test:
            load_path = os.path.join(self.save_model_folder, f"{self.save_model_name}_best_test.pkl")
            if not os.path.exists(load_path):
                self.logger.warning(f"Best test model checkpoint not found at {load_path}. Loading the best validation model instead.")
                load_path = os.path.join(self.save_model_folder, f"{self.save_model_name}.pkl")
        else:
            load_path = os.path.join(self.save_model_folder, f"{self.save_model_name}.pkl")

        self.logger.info(f"Loading model checkpoint from {load_path}")
        model.load_state_dict(torch.load(load_path))