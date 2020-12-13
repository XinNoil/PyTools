from basictorch.models.dnn import *
from torch.utils.data import RandomSampler

class SemiModel(DNN):
    def train_on_epoch(self):
        data_loader = self.datasets.get_train_loader()
        unlab_loader = self.datasets.get_unlab_loader(len(data_loader))
        for (b, batch_data_l),(b, batch_data_u) in zip(enumerate(data_loader), enumerate(unlab_loader)):
            batch_data = batch_data_l+batch_data_u
            losses = self.train_on_batch(b, batch_data)
            t.print_batch(self.epoch, self.epochs, b, self.batch_size, self.num_data, losses)

    def forward(self, x, x_u):
        return self.sequential(x)

    def get_losses(self, batch_data):
        inputs, labels, unlabs = batch_data
        outputs = self(inputs, unlabs)
        losses={}
        for r in self.loss_funcs:
            losses[r] = self.loss_funcs[r](outputs, labels)
        return losses
    
    def get_dataset_losses(self, dataset):
        with torch.no_grad():
            self.train_mode(False)
            return self.get_losses(tuple(dataset.tensors)+tuple(self.datasets.unlab_dataset.tensors))