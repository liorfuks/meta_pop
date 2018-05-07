import pickle
import os
import numpy as np


class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log(self, message, rank):
        with open(os.path.join(self.log_dir, "log_%d.txt" % rank), "a") as f:
            f.write(message+"\n")

    def write_general_stat(self, stat_string, rank):
        with open(os.path.join(self.log_dir, "stat_%d.txt" % rank), "a") as f:
            f.write(stat_string)

    def write_optimizer_stat(self, stat_string):
        if stat_string is not None:
            with open(os.path.join(self.log_dir, "optimizer_stat.txt"), "a") as f:
                f.write(stat_string)

    def save_parameters(self, parameters, iteration, rank):
        with open(os.path.join(self.log_dir, "parameters_%d_%d" % (rank, iteration)), 'wb') as f:
            pickle.dump({"parameters": parameters}, f)

    def save_vb(self, vb):
        np.save(os.path.join(self.log_dir, "vb.npy"), vb)
