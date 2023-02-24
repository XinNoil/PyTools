# -*- coding: UTF-8 -*-
class IModel():
    def __init__(self, cfg):
        super().__init__()

    def set_trainer(self, trainer):
        self.trainer = trainer

    def set_device(self, device):
        pass

    def train_mode(self):
        pass
    
    def valid_mode(self):
        pass

    def before_train(self):
        pass

    def before_epoch(self, epoch_id):
        pass
    
    
    # 训练Epoch
    def before_train_epoch(self, epoch_id):
        pass
    def before_train_batch(self, epoch_id, batch_id, batch_data):
        pass
    def train_batch(self, epoch_id, batch_id, batch_data):
        pass
    def after_train_batch(self, epoch_id, batch_id, batch_data):
        pass
    def after_train_epoch(self, epoch_id):
        pass

    # 验证Epoch
    def before_valid_epoch(self, epoch_id):
        pass
    def before_valid_batch(self, epoch_id, batch_id, batch_data):
        pass
    def valid_batch(self, epoch_id, batch_id, batch_data):
        pass
    def after_valid_batch(self, epoch_id, batch_id, batch_data):
        pass
    def after_valid_epoch(self, epoch_id):
        pass

    # 测试Epoch
    def before_test_epoch(self, epoch_id):
        pass
    def before_test_batch(self, epoch_id, batch_id, batch_data):
        pass
    def test_batch(self, epoch_id, batch_id, batch_data):
        pass
    def after_test_batch(self, epoch_id, batch_id, batch_data):
        pass
    def after_test_epoch(self, epoch_id):
        pass


    def after_epoch(self, epoch_id):
        pass

    def after_train(self):
        pass

    def predict(self, batch_data):
        pass
    
    def evaluate(self, test_trainer, suffix=None):
        pass

    def save(self, message="latest"):
        pass
    
    def load(self, model_path):
        pass


    
