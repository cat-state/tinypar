import os
import time
from typing import List, Tuple, Optional
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
    _reconfigure_microbatch_calculator,
)

import torch._dynamo

torch._dynamo.allow_in_graph(rearrange)


def identity(x):
    return x


# torch.compile = identity


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 1024  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


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


def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs_cis)


def reshape_for_broadcast(freqs, x_shape):
    ndim = len(x_shape)
    assert 0 <= 1 < ndim
    assert freqs.shape == (
        x_shape[1],
        x_shape[-2],
        x_shape[-1],
    ), f"{freqs.shape=} not compatible with {x_shape=}"
    shape = [d if i == 1 or i >= ndim - 2 else 1 for i, d in enumerate(x_shape)]
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
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        seqlen, bsz, _ = x.shape

        xq, xk, xv = add_bias(self.wq(x)), add_bias(self.wk(x)), add_bias(self.wv(x))
        xq = rearrange(xq, "s b (nh hd) -> b s nh hd", nh=self.n_local_heads)
        xk = rearrange(xk, "s b (nh hd) -> b s nh hd", nh=self.n_local_heads)
        values = rearrange(xv, "s b (nh hd) -> b s nh hd", nh=self.n_local_heads)

        xq, keys = apply_rotary_emb(xq, xk, freqs=freqs)

        keys = rearrange(keys, "b s nh hd -> b nh s hd")
        values = rearrange(values, "b s nh hd -> b nh s hd")
        xq = rearrange(xq, "b s nh hd -> b nh s hd")

        with torch.backends.cuda.sdp_kernel(
            enable_math=False, enable_flash=True, enable_mem_efficient=False
        ):
            output = F.scaled_dot_product_attention(xq, keys, values, is_causal=True)
            output = rearrange(output, "b nh s hd -> s b (nh hd)")
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
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class SplitLlama(nn.Module):
    def __init__(self, args: ModelArgs, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.pp_world = parallel_state.get_pipeline_model_parallel_world_size()
        self.tp_world = parallel_state.get_tensor_model_parallel_world_size()

        curr_rank_layers = args.n_layers // self.pp_world
        start_layer = self.pp_rank * curr_rank_layers

        self.layers = nn.ModuleList(
            [
                TransformerBlock(i + start_layer, args, dtype)
                for i in range(curr_rank_layers)
            ]
        )
        self.freqs = precompute_freqs(args.dim // args.n_heads, args.max_seq_len * 2)

        if self.pp_rank == 0:
            self.tok_embeddings = tensor_parallel.VocabParallelEmbedding(
                args.vocab_size, args.dim, params_dtype=dtype
            )

        if self.pp_rank == self.pp_world - 1:
            self.output = tensor_parallel.ColumnParallelLinear(
                args.dim, args.vocab_size, bias=False, params_dtype=dtype
            )
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.args = args

    # factored out for torch.compile
    @torch.compile
    def transformer_block(self, x, start_pos, freqs, mask):
        for layer in self.layers:
            x = layer(x, start_pos, freqs, mask)
        return x

    def forward(self, tokens_or_hidden_state: torch.Tensor, start_pos: int):
        if self.pp_rank == 0:
            x = self.tok_embeddings(tokens_or_hidden_state)
            x = rearrange(x, "b s d -> s b d")
        else:
            x = tokens_or_hidden_state

        seq_len, batch_size, _ = x.shape

        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(x)

        freqs = self.freqs[start_pos : start_pos + seq_len].to(x.device)
        head_dim = self.args.dim // self.args.n_heads
        q_shape = (batch_size, seq_len, self.args.n_heads, head_dim // 2, 2)
        freqs = reshape_for_broadcast(freqs, q_shape).to(x.device)

        x = self.transformer_block(x, start_pos, freqs, mask)

        if self.pp_rank == self.pp_world - 1:
            x = self.norm(x)
            x = add_bias(self.output(x))
            return x
        else:
            return x


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


def model_provider_func(llama_args, *args, **kwargs):
    return PipelineStage(SplitLlama(llama_args, dtype=torch.bfloat16))


def loss_func(pred, label):
    label = rearrange(label, "b s -> s b").contiguous()
    loss = tensor_parallel.vocab_parallel_cross_entropy(pred, label).mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"nice_loss": averaged_loss}


def train_forward_step_func(batch, model):
    input, label = batch
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    # if pp_rank == 0 and tp_rank == 0:
    #     print(parallel_state.get_data_parallel_rank(), input[0, :3], flush=True)
    # print(pp_rank, tp_rank, input.shape, label.shape, input.dtype, flush=True)
    out = model(input, start_pos=0)
    # print(pp_rank, tp_rank, out.shape, out.dtype, flush=True)
    return out.contiguous(), lambda pred: loss_func(pred.float(), label)


def inference_forward_step_func(batch, model):
    (input,) = batch
    out = model(input, start_pos=0)
    return out.contiguous(), lambda pred: (pred, {"logits": pred})


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
    7: ModelArgs(dim=4096, n_heads=32, n_layers=32, vocab_size=50432, norm_eps=1e-6),
}


def convert_llama_state_dict(
    args: ModelArgs,
    state_dict,
    tp_rank: int,
    tp_world: int,
    pp_rank: int,
    pp_world: int,
):
    state_dict = state_dict.copy()
    state_dict.pop("rope.freqs")
    # in original code, token embeddings are sharded across latent dim, but apex shards them along vocab dim
    if pp_rank == 0:
        tok_embeds = state_dict["tok_embeddings.weight"].cuda()
        full_embeds = tensor_parallel.gather_from_tensor_model_parallel_region(tok_embeds)
        local_vocab_size = args.vocab_size // tp_world
        tok_embeds = full_embeds[
            tp_rank * local_vocab_size : (tp_rank + 1) * local_vocab_size
        ]
        state_dict["tok_embeddings.weight"] = tok_embeds.cpu()
    else:
        state_dict.pop("tok_embeddings.weight")
    
    if pp_rank != (pp_world - 1):
        state_dict.pop("norm.weight")
        state_dict.pop("output.weight")
    
    def offset_layer_idx(name):
        stage_layers = args.n_layers // pp_world
        if name.startswith("layers."):
            layer_idx = int(name.split(".")[1])
            if pp_rank * stage_layers <= layer_idx < (pp_rank + 1) * stage_layers:
                new_layer_idx = layer_idx - pp_rank * stage_layers
                return name.replace(f"layers.{layer_idx}", f"layers.{new_layer_idx}")
            else:
                return None
        else:
            return name

    state_dict = {
        offset_layer_idx(k): v
        for k, v in state_dict.items()
        if offset_layer_idx(k) is not None
    }
    state_dict = {("module.wrapped." + k): v for k, v in state_dict.items()}
    return state_dict


from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os


logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


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

    tok = Tokenizer("/mnt/hdd/llama2/tokenizer.model")
    llama_args = ModelArgs(**dict(params[65].__dict__, vocab_size=tok.n_words))

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    state_dict = torch.load(f"/mnt/hdd/llama2/65B/consolidated.{tp_rank:02d}.pth")
    state_dict = convert_llama_state_dict(
        llama_args,
        state_dict,
        tp_rank,
        tensor_model_parallel_size,
        pp_rank,
        pipeline_model_parallel_size,
    )

    global_batch_size = 32
    micro_batch_size = 1

    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size * data_parallel_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )

    set_random_seed(2023)

    forward_backward_func = get_forward_backward_func(
        virtual_pipeline_model_parallel_size, pipeline_model_parallel_size
    )
    print(f"{forward_backward_func=}")

    model_kwargs = dict(llama_args=llama_args)
    wrap_with_ddp = True

    models = build_model(
        model_provider_func,
        wrap_with_ddp,
        virtual_pipeline_model_parallel_size,
        **model_kwargs,
    )

    models[0].load_state_dict(state_dict)
    print("loaded state dict", flush=True)

    local_rank = torch.cuda.current_device()

    optimizer = torch.optim.AdamW(models[0].parameters(), lr=1e-4)

    dp_rank = parallel_state.get_data_parallel_rank()

    data_loader = (
        torch.randint(
            0,
            llama_args.vocab_size,
            (100, global_batch_size, llama_args.max_seq_len + 1),
        )
        .long()
        .cuda()
    )

    data_loader = (
        torch.full(
            (100, global_batch_size, llama_args.max_seq_len + 1),
            fill_value=dp_rank,
        )
        .long()
        .cuda()
    )

    io_shape = (llama_args.max_seq_len, micro_batch_size, llama_args.dim)
    approx_model_flops = 8 * global_batch_size * data_parallel_size * llama_args.max_seq_len * 65e9

    if rank == 0:
        print(f"start {io_shape}", flush=True)

    prompt = [tok.encode("Hello world, my name is", bos=True, eos=False)]
    prompt_lengths = [len(p) for p in prompt]
    prompt = [p + [tok.eos_id] * (len(p) - llama_args.max_seq_len) for p in prompt]
    prompt = torch.tensor(prompt).long().cuda()

    # _reconfigure_microbatch_calculator(
    #     rank=rank,
    #     rampup_batch_size=None,
    #     global_batch_size=micro_batch_size,
    #     micro_batch_size=micro_batch_size,
    #     data_parallel_size=1,
    # )

    # with torch.no_grad():
    #     for i in range(100):
    #         output = forward_backward_func(
    #             inference_forward_step_func,
    #             [prompt],
    #             models,
    #             forward_only=True,
    #             tensor_shape=(prompt.shape[1], 1, llama_args.dim),
    #             dtype=torch.bfloat16,
    #         )

    #         if parallel_state.is_pipeline_last_stage():
    #             logits = output[0]["logits"].float()
    #             logits = rearrange(logits, "s b n -> b s n")
    #             logits = tensor_parallel.gather_from_tensor_model_parallel_region(
    #                 logits
    #             )
    #             prompt = torch.cat([prompt, logits[:, -1:].argmax(dim=-1)], dim=1)
    #             src = parallel_state.get_pipeline_model_parallel_last_rank()
    #             group = parallel_state.get_embedding_group()
    #             torch.distributed.broadcast(prompt, src, group)
    #         elif parallel_state.is_pipeline_first_stage():
    #             new_prompt = torch.empty(
    #                 (prompt.shape[0], prompt.shape[1] + 1),
    #                 dtype=prompt.dtype,
    #                 device=prompt.device,
    #             )
    #             src = parallel_state.get_pipeline_model_parallel_last_rank()
    #             group = parallel_state.get_embedding_group()
    #             torch.distributed.broadcast(new_prompt, src, group)
    #             prompt = new_prompt

    #         if rank == 0:
    #             text_output = tok.decode(prompt[0].cpu().numpy().tolist())
    #             print(text_output)

    # return
    # _reconfigure_microbatch_calculator(
    #     rank=rank,
    #     rampup_batch_size=None,
    #     global_batch_size=global_batch_size,
    #     micro_batch_size=micro_batch_size,
    #     data_parallel_size=data_parallel_size,
    # )

    for batch in data_loader:
        optimizer.zero_grad()
        inputs, labels = batch[:, :-1], batch[:, 1:]
        t = time.time()
        loss = forward_backward_func(
            train_forward_step_func,
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
            samples_per_sec = data_parallel_size * global_batch_size / dt
            print(f"throughput: {samples_per_sec=}", flush=True)
            print(f"{len(loss)=}", flush=True)
        optimizer.step()

    print("done", flush=True)


if __name__ == "__main__":
    main()
