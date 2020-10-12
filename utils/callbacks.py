from catalyst.dl.core import Callback, CallbackOrder, CallbackNode
from typing import List, Dict
from comet_ml import Experiment


class CometCallback(Callback):
    """
    Logger callback for Comet.ml
    """

    def __init__(
            self,
            experiment: Experiment = None ,
            metric_names: List[str] = None,
            log_on_batch_end : bool = True,
            log_on_epoch_end : bool = True
    ):

        super().__init__(order=CallbackOrder.Logging, node=CallbackNode.Master)
        self.experiment = experiment
        self.matrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        if not (self.log_on_batch_end or self.log_on_epoch_end):
            raise ValueError("Comet logger has nothing to log!")

    def on_stage_start(self, state: "State"):
        assert self.experiment is not None

    def on_batch_end(self, state: "State"):

        if self.matrics_to_log is None:
            metrics = state.batch_metrics
        else:
            metrics = self.matrics_to_log

        if self.log_on_batch_end:
            mode = state.loader_name
            prefix = mode + "/batch/"
            step = state.global_step
            self.experiment.log_metrics(metrics, prefix=prefix, step=step)

    def on_epoch_start(self, state: "State"):

        epoch = state.global_epoch
        self.experiment.log_current_epoch(epoch)

    def on_epoch_end(self, state: "State"):

        if self.matrics_to_log is None:
            metrics = state.epoch_metrics
        else:
            metrics = self.matrics_to_log

        if self.log_on_batch_end:
            mode = state.loader_name
            prefix = "epoch/"
            epoch = state.global_epoch
            self.experiment.log_metrics(metrics, prefix=prefix,step=epoch)



