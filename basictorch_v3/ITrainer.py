# -*- coding: UTF-8 -*-

class ITrainer():
    def __init__(self, cfg):
        super().__init__()

    def before_train(self):
        pass

    def train(self):
        pass

    def after_train(self):
        pass

    def save_losses(self):
        pass

    def plot_losses(self):
        pass