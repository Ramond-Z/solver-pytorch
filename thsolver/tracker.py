# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import time
import torch
import torch.distributed
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional


class AverageTracker:
    r"""Tracks and logs averaged scalar tensors across iterations and epochs."""

    def __init__(self, synchronize_cuda: bool = True):
        r"""Initializes the tracker state."""

        self.value = dict()
        self.num = dict()
        self.max_len = 76
        self.synchronize_cuda = bool(synchronize_cuda)
        self.start_time = self.get_time()
        self.last_time = self.start_time

    def get_time(self):
        r"""Returns the current synchronized wall-clock time."""

        if self.synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    def update(self, value: Dict[str, torch.Tensor]):
        r"""Update the tracker with the given value. This function is called at the
        end of each iteration.
        """

        for key, val in value.items():
            val = val.detach() if torch.is_tensor(val) else torch.as_tensor(val)
            if torch.is_tensor(val) and not torch.isfinite(val).all():
                if key not in self.value:
                    self.value[key] = torch.zeros_like(val)
                    self.num[key] = 0
                continue
            self.value[key] = self.value.get(key, 0) + val
            self.num[key] = self.num.get(key, 0) + 1

    def record_time(self, num_iters: int = 1):
        r"""Roughly records the elapsed time per iteration.

        Args:
          num_iters (int): The number of iterations represented by the update.
        """

        curr_time = self.get_time()
        elapsed = curr_time - self.last_time
        self.value["time/iter"] = self.value.get("time/iter", 0) + elapsed
        self.num["time/iter"] = self.num.get("time/iter", 0) + num_iters
        self.last_time = curr_time

    def average(self):
        r"""Returns the averaged values accumulated in the tracker."""

        return {
            key: float(val) / self.num[key]
            for key, val in self.value.items()
            if self.num[key] > 0
        }

    @torch.no_grad()
    def average_all_gather(self):
        r"""Average the tensors on all GPUs using all_gather, which is called at the
        end of each epoch.
        """

        for key, tensor in self.value.items():
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                # only gather tensors on GPU
                values_gather = [
                    torch.ones_like(tensor)
                    for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather(values_gather, tensor, async_op=False)
                values = torch.stack(values_gather, dim=0)

                count = torch.tensor(
                    float(self.num[key]),
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                counts_gather = [
                    torch.ones_like(count)
                    for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather(counts_gather, count, async_op=False)
                counts = torch.stack(counts_gather, dim=0)

                self.value[key] = values.sum(dim=0)
                self.num[key] = int(counts.sum().item())

    def log(
        self,
        epoch: int,
        summary_writer: Optional[SummaryWriter] = None,
        log_file: Optional[str] = None,
        msg_tag: str = "->",
        notes: str = "",
        print_time: bool = True,
        print_memory: bool = False,
        print_console: bool = True,
    ):
        r"""Logs the average value to the console, TensorBoard, and a log file.

        Args:
          epoch (int): The current epoch index.
          summary_writer (SummaryWriter or None): The TensorBoard writer.
          log_file (str or None): The CSV-like log file path.
          msg_tag (str): The prefix printed before the log line.
          notes (str): Extra notes appended to the log message.
          print_time (bool): If True, prints the timestamp and elapsed time.
          print_memory (bool): If True, prints the reserved CUDA memory.
        """

        avg = self.average()
        msg = "Epoch: %d" % epoch
        for key, val in avg.items():
            msg += ", %s: %.2e" % (key, val)
            if summary_writer is not None:
                summary_writer.add_scalar(key, val, epoch)

        # if the log_file is provided, save the log
        if log_file is not None:
            with open(log_file, "a") as fid:
                fid.write(msg + "\n")

        # memory
        memory = ""
        if print_memory and torch.cuda.is_available():
            size = torch.cuda.memory_reserved()
            # size = torch.cuda.memory_allocated()
            memory = ", memory: {:.3f}GB".format(size / 2**30)

        # time
        time_str = ""
        if print_time:
            curr_time = self.get_time()
            time_str += ", time: " + datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            time_str += ", duration: {:.2f}s".format(curr_time - self.start_time)

        # other notes
        if notes:
            notes = ", " + notes

        # concatenate all messages
        msg += memory + time_str + notes

        # split the msg for better display
        chunks = [msg[i : i + self.max_len] for i in range(0, len(msg), self.max_len)]
        msg = (msg_tag + " ") + ("\n" + len(msg_tag) * " " + " ").join(chunks)
        if print_console:
            tqdm.write(msg)
