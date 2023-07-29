# basictorch_v3
## BaseModel
### Description
The basic model encapsulates Pytorch.

### Usage
```python
from basictorch_v3 import BaseModel

class YourModel(BaseModel):
    def __init__(self, *params):
        super.__init__(*params)
        pass

    # at least override function 
    def train_step(self, epoch_id, batch_id, batch_data):
        pass

    # at least override function 
    def valid_step(self, epoch_id, batch_id, batch_data):
        pass

    # other functions
```

### Class
| function                           | description                                                          |
| ---------------------------------- | -------------------------------------------------------------------- |
| `__init__`                         | Initiate model config, optimizer, scheduler, loss_function           |
| `create_optimizer`                 | Create Optimizer with config.optimizer                               |
| `create_scheduler`                 | Create Scheduler with config.scheduler for Optimizer                 |
| `create_loss_function`             | Create Loss_Function with config.loss_function                       |
| `make_necessary_dirs`              | Create dirs, e.g., model, figure, and evaluation for later use       |
| `save_on_metrics`                  | Register epoch end metrics on dict structure, e.g., valid loss and test error |
| `evaluate_on_metrics`              | Record evaluate metrics on list structure, e.g., valid loss and test error |
| `device`                           | Get current device                                                   |
| `set_device`                       | Set the network device                                               |
| `train_mode`                       | Set the network on train mode                                        |
| `valid_mode`                       | Set the network on eval mode                                         |
| `before_epoch`                     | Called before each epoch, save model at interval point               |
| `after_epoch`                      | Output epoch metrics, update scheduler and save model                |
| `train_step`                       | Train on batch data, need to override it                             |
| `valid_step`                       | Valid on batch data                                                  |
| `test_step`                        | Test on batch data, need to override it                              |
| `train_batch`                      | Call function 'train_step', log train loss, backward gradient        |
| `valid_batch`                      | Call function 'valid_step', log valid loss                           |
| `test_batch`                       | Call function 'test_step', log test error                            |
| `after_train`                      | Plot losses figure, save model and save epoch metrics to csv file    |
| `plot_losses`                      | Plot losses figure, e.g., Train Loss, Valid Loss, Test Error         |
| `evaluate_step`                    | Evaluate test dataset                                                |
| `evaluate`                         | Call function 'evaluate_step', evalute metrics, e.g., valid loss and test error |
| `save`                             | Save model, e.g., net, optimizer, scheduler state dict to pt file    |
| `load`                             | Load model, e.g., net, optimizer, scheduler state dict to device     |
| `get_optimizer_parameters`         | Get net parameters                                                   |
| `log_epoch_metrics`                | Log metrics by name in batch data level                              |
| `log_epoch_metrics_dict`           | Log multi-metrics, class function 'log_epoch_metrics'                |
| `log_history_metrics_dict`         | Log metrics by name in epoch level                                   |
| `print_epoch_metrics`              | Print latest epoch metrics                                           |
| `get_epoch_metrics`                | Get epoch metrics values by id                                       |
| `get_epoch_metrics_history`        | Get epoch metrics epoch id and values by name                        |
| `save_epoch_metrics`               | Save epoch metrics to csv file                                       |
| `save_model_on_epoch_end`          | Call function 'save' on epoch end when values attr is better than old|
| `update_scheduler`                 | Update scheduler after epoch                                         |

## BaseTrainer
### Description
The basic trainer encapsulates BaseModel.

### Usage
```python
from basictorch_v3 import BaseTrainer

class YourTrainer(BaseTrainer):
    def __init__(self, *params):
        super.__init__(*params)
        pass

    # other functions
```

### Class
| function                           | description                                                          |
| ---------------------------------- | -------------------------------------------------------------------- |
| `__init__`                         | Initiate trainer config, model, and train/valid/test dataloader      |
| `train`                            | Train data number of epoch times                                     |
| `before_train`                     | Set train manual seed, device and process bar                        |
| `after_train`                      | Call function 'model.after_train' and close process bar              |
| `epoch_step`                       | Train data one epoch time                                            |
| `before_epoch`                     | Call function 'model.before_epoch'                                   |
| `train_epoch`                      | Train on train dataloader, call function 'model.train_batch' when loop |
| `valid_epoch`                      | Valid on valid dataloader, call function 'model.valid_batch' when loop |
| `test_epoch`                       | Test on test dataloader,  call function 'model.test_batch' when loop |
| `after_epoch`                      | Call function 'model.after_epoch'                                    |
| `on_train_interrupted`             | Train interrupted by keyboard                                        |   