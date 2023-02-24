from lightning.fabric.loggers import CSVLogger, TensorBoardLogger, Logger
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars, convert_to_tensors
from typing import Any, Callable, cast, Dict, Generator, List, Mapping, Optional, overload, Sequence, Tuple, Union

# https://pytorch-lightning.readthedocs.io/en/stable/fabric/guide/logging.html

# tb_logger = TensorBoardLogger(root_dir="logs/tensorboard")
# csv_logger = CSVLogger(root_dir="logs/csv")

# # Add multiple loggers in a list
# logger = Logger(loggers=[tb_logger, csv_logger])

# # Calling .log() or .log_dict() always logs to all loggers simultaneously
# fabric.log("some_value", value)

class Logger():
    def __init__(self, loggers: Optional[Union[Logger, List[Logger]]] = None) -> None:
        loggers = loggers if loggers is not None else []
        self._loggers = loggers if isinstance(loggers, list) else [loggers]
    
    def log(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Log a scalar to all loggers that were added to Fabric.

        Args:
            name: The name of the metric to log.
            value: The metric value to collect. If the value is a :class:`torch.Tensor`, it gets detached from the
                graph automatically.
            step: Optional step number. Most Logger implementations auto-increment the step value by one with every
                log call. You can specify your own value here.
        """
        self.log_dict(metrics={name: value}, step=step)

    def log_dict(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Log multiple scalars at once to all loggers that were added to Fabric.

        Args:
            metrics: A dictionary where the key is the name of the metric and the value the scalar to be logged.
                Any :class:`torch.Tensor` in the dictionary get detached from the graph automatically.
            step: Optional step number. Most Logger implementations auto-increment this value by one with every
                log call. You can specify your own value here.
        """
        metrics = convert_tensors_to_scalars(metrics)
        for logger in self._loggers:
            logger.log_metrics(metrics=metrics, step=step)