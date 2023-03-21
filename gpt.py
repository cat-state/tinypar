import os
import time
from typing import List, Tuple, Optional
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from socket import gethostname

from einops import rearrange

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
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam

import torch._dynamo

torch._dynamo.allow_in_graph(rearrange)
# torch._dynamo.allow_in_graph(
#     tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce
# )


class PipelineStage(nn.Module):
    input_tensors: Optional[List[torch.Tensor]] = None

    def __init__(self, module):
        super().__init__()
        self.input_tensors = None
        self.wrapped = module

    def set_input_tensor(self, tensor: List[torch.Tensor]):
        self.input_tensors = tensor

    def forward(self, *x, **kwargs):
        if parallel_state.is_pipeline_first_stage():
            inputs = x
        else:
            inputs = self.input_tensors
        return self.wrapped(*inputs, **kwargs)


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 1024  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 4096


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.float32):
        super().__init__()
        self.eps = torch.tensor(eps)
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    @torch.compile
    def _norm(self, x, eps, weight):
        out = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).type_as(x)
        return out * weight

    def forward(self, x):
        return self._norm(x, self.eps, self.weight)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), f"{freqs_cis.shape=} != {x.shape=}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


"rewrite above to not use complex numbers, using cmul instead"


def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return torch.view_as_real(freqs_cis)


def reshape_for_broadcast(freqs, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs.shape == (
        x.shape[1],
        x.shape[-2],
        x.shape[-1],
    ), f"{freqs.shape=} != {x.shape=}"
    shape = [d if i == 1 or i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs.view(*shape)


def cmul(x, y):
    return torch.stack(
        [
            x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1],
            x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0],
        ],
        dim=-1,
    )


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_out = cmul(xq_, freqs).flatten(3)
    xk_out = cmul(xk_, freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# torch._dynamo.allow_in_graph(apply_rotary_emb)


def add_bias(x: Tuple[torch.tensor, Optional[torch.Tensor]]):
    x, bias = x
    if bias is not None:
        x = x + bias
    return x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, dtype: torch.dtype = torch.float32):
        super().__init__()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert args.n_heads % tp_size == 0
        self.n_local_heads = args.n_heads // tp_size
        self.head_dim = args.dim // args.n_heads

        self.wq = tensor_parallel.ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            params_dtype=dtype,
        )
        self.wk = tensor_parallel.ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            params_dtype=dtype,
        )
        self.wv = tensor_parallel.ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            params_dtype=dtype,
        )
        self.wo = tensor_parallel.RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            params_dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        seqlen, bsz, _ = x.shape

        xq, xk, xv = add_bias(self.wq(x)), add_bias(self.wk(x)), add_bias(self.wv(x))
        xq = rearrange(xq, "s b (nh hd) -> b s nh hd", nh=self.n_local_heads)
        xk = rearrange(xk, "s b (nh hd) -> b s nh hd", nh=self.n_local_heads)
        values = rearrange(xv, "s b (nh hd) -> b s nh hd", nh=self.n_local_heads)

        xq, keys = apply_rotary_emb(xq, xk, freqs=freqs_cis)

        keys = rearrange(keys, "b s nh hd -> b nh s hd")
        values = rearrange(values, "b s nh hd -> b nh s hd")
        xq = rearrange(xq, "b s nh hd -> b nh s hd")

        if True:
            with torch.backends.cuda.sdp_kernel(
                enable_math=False, enable_flash=True, enable_mem_efficient=False
            ):
                output = F.scaled_dot_product_attention(
                    xq, keys, values, is_causal=True
                )
                output = rearrange(output, "b nh s hd -> s b (nh hd)")
                return add_bias(self.wo(output))

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = rearrange(output, "b s h -> s b h")
        return add_bias(self.wo(output))


@torch.compile
def gated_silu(x, gate):
    return F.silu(x) * gate


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = tensor_parallel.ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            params_dtype=dtype,
        )
        self.w2 = tensor_parallel.RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            params_dtype=dtype,
        )
        self.w3 = tensor_parallel.ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            params_dtype=dtype,
        )

    def forward(self, x):
        return add_bias(self.w2(gated_silu(add_bias(self.w1(x)), add_bias(self.w3(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, dtype: torch.dtype):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, dtype=dtype)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dtype=dtype,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class SplitLlama(nn.Module):
    def __init__(self, args: ModelArgs, dtype: torch.dtype = torch.float32):
        super().__init__()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_world = parallel_state.get_pipeline_model_parallel_world_size()
        tp_world = parallel_state.get_tensor_model_parallel_world_size()

        curr_rank_layers = args.n_layers // pp_world
        start_layer = pp_rank * curr_rank_layers

        self.layers = nn.ModuleList(
            [
                torch.compile(TransformerBlock(i + start_layer, args, dtype))
                for i in range(curr_rank_layers)
            ]
        )
        self.freqs_cis = reshape_for_broadcast(
            precompute_freqs(args.dim // args.n_heads, args.max_seq_len),
            torch.empty(
                (
                    1,
                    args.max_seq_len,
                    args.n_heads // tp_world,
                    args.dim // args.n_heads // 2,
                    2,
                ),
                dtype=dtype,
            ),
        )

        if pp_rank == 0:
            self.tok_embeddings = tensor_parallel.VocabParallelEmbedding(
                args.vocab_size, args.dim, params_dtype=dtype
            )

        if pp_rank == pp_world - 1:
            self.output = tensor_parallel.ColumnParallelLinear(
                args.dim, args.vocab_size, bias=False, params_dtype=dtype
            )
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.pp_rank = pp_rank
        self.pp_world = pp_world

    def forward(self, tokens_or_hidden_state: torch.Tensor, start_pos: int):
        if self.pp_rank == 0:
            x = self.tok_embeddings(tokens_or_hidden_state)
            x = rearrange(x, "b s d -> s b d")
        else:
            x = tokens_or_hidden_state

        seq_len, batch_size, _ = x.shape
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(x)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len].to(x.device)
        for layer in self.layers:
            x = layer(x, start_pos, freqs_cis, mask)

        if self.pp_rank == self.pp_world - 1:
            x = self.norm(x)
            x = add_bias(self.output(x))
            return x
        else:
            return x


def model_provider_func(llama_args, *args, **kwargs):
    return PipelineStage(SplitLlama(llama_args, dtype=torch.bfloat16))


def loss_func(pred, label):
    label = rearrange(label, "b s -> s b").contiguous()

    loss = tensor_parallel.vocab_parallel_cross_entropy(pred, label).mean()

    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"nice_loss": averaged_loss}


def forward_step_func(batch, model):
    input, label = batch
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    # print(pp_rank, tp_rank, input.shape, label.shape, input.dtype, flush=True)
    out = model(input, start_pos=0)
    # print(pp_rank, tp_rank, out.shape, out.dtype, flush=True)
    return out.contiguous(), lambda pred: loss_func(pred.float(), label)


# from apex
def set_random_seed(seed: int):
    """Set random seed for reproducability."""
    # Ensure that different pipeline MP stages get different seeds.
    # TP seeds are automatically offset by the TP rank by apex.

    seed = seed + (100 * parallel_state.get_pipeline_model_parallel_rank())
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed)


params = {
    65: ModelArgs(dim=8192, n_heads=64, n_layers=80, vocab_size=50432, norm_eps=1e-5),
    # 30: ModelArgs(dim=6656, n_heads=52, n_layers=60, vocab_size=50432, norm_eps=1e-6),
    30: ModelArgs(dim=8192, n_heads=64, n_layers=40, vocab_size=50432, norm_eps=1e-6),
    15: ModelArgs(dim=8192, n_heads=64, n_layers=20, vocab_size=50432, norm_eps=1e-6),
}

# import logging
# logging.basicConfig(level=logging.DEBUG)


def main():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"hi from {rank}/{world_size} on {gethostname()}", flush=True)

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    tensor_model_parallel_size = 8
    pipeline_model_parallel_size = 2
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

    global_batch_size = 32
    micro_batch_size = 1

    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )

    set_random_seed(2023)

    forward_backward_func = get_forward_backward_func(
        virtual_pipeline_model_parallel_size, pipeline_model_parallel_size
    )
    print(f"{forward_backward_func=}")

    llama_args = params[65]
    model_kwargs = dict(llama_args=llama_args)
    wrap_with_ddp = True

    models = build_model(
        model_provider_func,
        wrap_with_ddp,
        virtual_pipeline_model_parallel_size,
        **model_kwargs,
    )

    local_rank = torch.cuda.current_device()

    # optimizer = DistributedFusedAdam(
    #     models[0].parameters(),
    #     lr=1e-4,
    #     process_group=parallel_state.get_data_parallel_group(),
    #     store_params=False,  # disable sharded param store
    # )

    optimizer = torch.optim.AdamW(models[0].parameters(), lr=1e-4)

    data_loader = (
        torch.randint(
            0,
            llama_args.vocab_size,
            (100, global_batch_size, llama_args.max_seq_len + 1),
        )
        .long()
        .cuda()
    )

    io_shape = (llama_args.max_seq_len, micro_batch_size, llama_args.dim)
    approx_model_flops = 8 * global_batch_size * llama_args.max_seq_len * 65e9

    if rank == 0:
        print(f"start {io_shape}", flush=True)

    for batch in data_loader:
        optimizer.zero_grad()
        inputs, labels = batch[:, :-1], batch[:, 1:]
        t = time.time()
        loss = forward_backward_func(
            forward_step_func,
            [inputs, labels],
            models,
            forward_only=False,
            tensor_shape=io_shape,
            dtype=torch.bfloat16,
        )

        dt = time.time() - t
        if rank == (world_size - 1):
            print(
                f"tflops: {approx_model_flops / (dt * world_size) / 1e12=}", flush=True
            )
            memory_usage_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"memory usage: {memory_usage_gb=}", flush=True)
            print(f"{loss=}", flush=True)
        optimizer.step()

    print("done", flush=True)


if __name__ == "__main__":
    main()
