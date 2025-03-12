import os
import torch.utils.tensorboard as tb
from cf.tools.string_tools import now_string

# Log training progress to tensorboard
class TrainingLog:

    def __init__(self, log_dir, add_unique_str=True):
        if add_unique_str:
            log_dir = os.path.join(log_dir, now_string())
        self.writer = tb.SummaryWriter(log_dir)

    # make sure that data is sent to tensorboard
    def flush(self):
        self.writer.flush()

    # close the logger
    def close(self):
        self.writer.close()
    
    # add a scalar
    def add_scalar(self, name, value, step, walltime=None):
        self.writer.add_scalar(name, value, global_step=step, walltime=walltime)

    # add an image
    def add_image(self, name, image, step=0, force_flush=True):
        self.writer.add_image(name, image, step, dataformats='HWC')
        if force_flush:
            self.flush()

    # add a graph. (The graph can be a neural network)
    def add_graph(self, graph, graph_input):
        self.writer.add_graph(graph, graph_input)

