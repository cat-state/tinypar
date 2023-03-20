import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from math import sqrt
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from apex.transformer import parallel_state
from apex.transformer import tensor_parallel
from apex.transformer.pipeline_parallel import get_forward_backward_func, build_model
from apex.transformer.pipeline_parallel.utils import (
    average_losses_across_data_parallel_group,
    setup_microbatch_calculator,
)


class PipelineStage(nn.Module):
    input_tensors: Optional[List[torch.Tensor]] = None

    def __init__(self, module):
        super().__init__()
        self.input_tensors = None
        self.wrapped = module

    def set_input_tensor(self, tensor: List[torch.Tensor]):
        self.input_tensors = tensor

    def forward(self, x):
        if parallel_state.is_pipeline_first_stage():
            inputs = [x]
        else:
            inputs = self.input_tensors
        return self.wrapped(*inputs)


class ParallelLinear(nn.Module):
    """Linear layer parallelized over the longer dimension."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        init_method=partial(nn.init.kaiming_uniform_, a=sqrt(5), nonlinearity="relu"),
        use_cpu_initialization=False,
        bias=True,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        gather_output=True,
        input_is_parallel=False,
    ):
        super().__init__()

        no_async_tensor_model_parallel_allreduce = (
            parallel_state.get_tensor_model_parallel_world_size() == 1
            or sequence_parallel
        )

        if in_size < out_size:
            self.layer = tensor_parallel.ColumnParallelLinear(
                in_size,
                out_size,
                gather_output=gather_output,
                init_method=init_method,
                skip_bias_add=False,
                use_cpu_initialization=use_cpu_initialization,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )
        else:
            self.layer = tensor_parallel.RowParallelLinear(
                in_size,
                out_size,
                input_is_parallel=input_is_parallel,
                init_method=init_method,
                skip_bias_add=False,
                use_cpu_initialization=use_cpu_initialization,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )

    def forward(self, x):
        output, bias = self.layer(x)
        if bias is not None:
            return output + bias
        return output


def make_parallel_mlp(n_embd: int, out: int, sequence_parallel=False) -> nn.Sequential:
    """Returns a generic sequential model parallel MLP head."""
    parallel_intermediate = out < (n_embd * 2)
    return nn.Sequential(
        ParallelLinear(
            n_embd,
            n_embd * 2,
            sequence_parallel=sequence_parallel,
            gather_output=not parallel_intermediate,
        ),
        nn.ReLU(),
        ParallelLinear(
            n_embd * 2,
            out,
            sequence_parallel=sequence_parallel,
            input_is_parallel=parallel_intermediate,
        ),
    )


def model_provider_func(*args, **kwargs):
    return PipelineStage(make_parallel_mlp(10, 10))


def loss_func(pred, label):
    loss = (pred - label) ** 2
    loss = loss.mean()
    print(f"{loss=} {torch.distributed.get_rank()=}")

    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"nice_loss": averaged_loss}


def forward_step_func(batch, model):
    input, label = batch
    out = model(input)
    print(out)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    return out, lambda pred: loss_func(pred, label)


# from
def set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * parallel_state.get_pipeline_model_parallel_rank())
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed_))


def main():
    print(
        f"{os.environ['LOCAL_RANK']=} {os.environ['RANK']=} {os.environ['WORLD_SIZE']=}"
    )
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file:///mnt/nvme/home/uwu/nccl.sock",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    print(f"{torch.distributed.get_rank()=}")

    tensor_model_parallel_size = 2
    pipeline_model_parallel_size = 4
    virtual_pipeline_model_parallel_size = None

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size,
    )

    world_size = torch.distributed.get_world_size()
    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size
    )
    setup_microbatch_calculator(
        rank=int(os.environ["RANK"]),
        rampup_batch_size=None,
        global_batch_size=8,
        micro_batch_size=1,
        data_parallel_size=data_parallel_size,
    )

    set_random_seed(2023)

    forward_backward_func = get_forward_backward_func(
        virtual_pipeline_model_parallel_size, pipeline_model_parallel_size
    )
    print(f"{forward_backward_func=}")
    model_kwargs = {}
    wrap_with_ddp = False

    models = build_model(
        model_provider_func,
        wrap_with_ddp,
        virtual_pipeline_model_parallel_size,
        **model_kwargs,
    )
    """assert len(models) == virtual_pipeline_model_parallel_size
    """
    local_rank = torch.cuda.current_device()
    models = [
        DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            process_group=parallel_state.get_data_parallel_group(),
        )
        for model in models
    ]

    optimizer = torch.optim.SGD(models[0].parameters(), lr=0.01)
    data_loader = torch.rand(100, 8, 16, 10).cuda()

    # io_shape = (micro_batch_size, seq_len, hidden_size)
    io_shape = (1, 16, 10)

    for batch in data_loader:
        loss = forward_backward_func(
            forward_step_func,
            [batch, torch.zeros_like(batch)],
            models,
            forward_only=False,
            tensor_shape=io_shape,
        )
        print(f"{loss=}", flush=True)
        optimizer.step()
        print("opt", flush=True)

    print("done")


if __name__ == "__main__":
    main()
