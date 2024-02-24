import gc
import math
from time import time
from argparse import ArgumentParser
import random
import wandb
import torch
from torch import nn
import matplotlib.pyplot as plt

from einops import rearrange
from torch.utils.cpp_extension import load, load_inline
from kac import KacRandomWalk_, kac_random_walk_, backward_kac_random_walk_, parametric_kac_random_walk_

from tqdm import tqdm

import transformers
from transformers.models.llama.modeling_llama import LlamaAttention
import datasets

ds = datasets.load_dataset("yahma/alpaca-cleaned")
print(ds)

def prompt(x):
    if x['input'] != '':
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{x['instruction']}

### Input:
{x['input']}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{x['instruction']}

### Response:"""
    
print(prompt(ds['train'][0]))

tokenizer = transformers.AutoTokenizer.from_pretrained("huggyllama/llama-7b")

args = ArgumentParser()
args.add_argument("--model", type=str, default="huggyllama/llama-7b")
args.add_argument("--batch_size", type=int, default=8)
args.add_argument("--lr", type=float, default=3e-4)
args.add_argument("--warmup_steps", type=int, default=600)
args.add_argument("--total_steps", type=int, default=None)
args.add_argument("--bf16", action="store_true")
args.add_argument("--accum_steps", type=int, default=1)
args.add_argument("--seed", type=int, default=2024)
args.add_argument("--device", type=int, default=0)


args = args.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device(f"cuda:{args.device}")
torch.set_default_device(device)

batch_size = args.batch_size

class KacLayer(nn.Module):
    def __init__(self, layer, dim, seed=2024, n_steps=None, adapter_dtype=torch.float32):
        super().__init__()
        inner_w = next(layer.parameters())
        self.vec = nn.Parameter(torch.zeros(dim, device=inner_w.device, dtype=adapter_dtype)/math.sqrt(dim))
        layer.requires_grad_(False)
        self.layer = layer
        self.dim = dim
        self.seed = seed
        self.n_steps = n_steps or math.ceil(math.log2(dim) * 0.3) * dim

    def forward(self, x):
        y = self.layer(x)
        x_2d = x.view(-1, self.dim).to(self.vec.dtype).clone()
        y_p = kac_random_walk_(x_2d, self.seed * 2 , self.n_steps)
        y_p = self.vec * y_p.to(self.vec.dtype)
        y_p = kac_random_walk_(y_p, self.seed * 2 + 1, self.n_steps)
        y_p = y_p.view_as(y).to(y.dtype)
        return y + y_p


class ParametricKacLayer(nn.Module):
    def __init__(self, dim, seed=2024, n_steps=None, adapter_dtype=torch.float32):
        super().__init__()
        self.angles = nn.Parameter(torch.zeros(dim, dtype=adapter_dtype))
        self.dim = dim
        self.seed = seed
        self.n_steps = n_steps or math.ceil(math.log2(dim) * 0.3) * dim

    def forward(self, x):
        x_2d = x.view(-1, self.dim).to(self.vec.dtype).clone()
        y_2d = parametric_kac_random_walk_(x_2d, self.angles, self.seed * 2, self.n_steps)
        y = y_2d.view_as(x).to(x.dtype)
        return y

class PrecomposedKacLayer(nn.Module):
    def __init__(self, layer: nn.Module, kac_layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.kac_layer = kac_layer

    def forward(self, x):
        return self.layer(self.kac_layer(x))



model = transformers.AutoModelForCausalLM.from_pretrained(args.model, load_in_8bit=not args.bf16, torch_dtype=torch.bfloat16, device_map=device)
print(model)

def replace_layer(layer_idx, m, dim, seed=2024):
    if hasattr(m, "__wrapped__"):
        return
    if isinstance(m, LlamaAttention):
        make_layer = lambda seed: KacLayer(dim, seed)
        m.q_proj = PrecomposedKacLayer(m.q_proj, make_layer(layer_idx * 4 + 1 + seed))
        m.k_proj = PrecomposedKacLayer(m.k_proj, make_layer(layer_idx * 4 + 2 + seed))
        m.v_proj = PrecomposedKacLayer(m.v_proj, make_layer(layer_idx * 4 + 3 + seed))
        m.o_proj = PrecomposedKacLayer(m.o_proj, make_layer(layer_idx * 4 + 4 + seed))
        m.__wrapped__ = True

def replace_layer_versa(layer_idx, m, dim, seed=2024):
    if hasattr(m, "__wrapped__"):
        return
    if isinstance(m, LlamaAttention):
        m.q_proj = KacLayer(m.q_proj, dim, layer_idx * 4 + 1 + seed)
        m.k_proj = KacLayer(m.k_proj, dim, layer_idx * 4 + 2 + seed)
        m.v_proj = KacLayer(m.v_proj, dim, layer_idx * 4 + 3 + seed)
        m.o_proj = KacLayer(m.o_proj, dim, layer_idx * 4 + 4 + seed)
        m.__wrapped__ = True

for name, module in model.named_modules():
    if "norm" in name:
        module.to(torch.float32)
    if "lm_head" in name:
        if hasattr(module, "weight"):
            module.to(torch.bfloat16)

for p in model.parameters():
    p.requires_grad_(False)

for i, l in enumerate(model.model.layers):
    replace_layer(i, l.self_attn, seed=args.seed, dim=4096)


accum_steps = args.accum_steps
warmup_steps = args.warmup_steps
total_steps = len(ds['train']) // batch_size 

def cosine_lr_with_warmup(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return 0.5 * (1 + math.cos((step - warmup_steps) / (total_steps - warmup_steps) * math.pi))

target_lr = args.lr

config = dict(
    batch_size=batch_size,
    accum_steps=accum_steps,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    target_lr=target_lr,
    model_name=args.model,
    seed=args.seed,
)

wandb.init(project="kac-llama", entity="uwu1", name="pre-kac-llama-7b", config=config)


t = time()

int8 = not args.bf16
if not int8:
    model.to(device)

optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
print(model)

all_idxs = [i for i in range(len(ds['train']))]
random.shuffle(all_idxs)
eval_idxs = all_idxs[:len(ds['train'])//10]
idxs = all_idxs[len(ds['train'])//10:]


for i in tqdm(range(len(idxs) // batch_size)):
    lr = target_lr * cosine_lr_with_warmup(i, warmup_steps, total_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


    ps = ["<s> " + prompt(ds['train'][idxs[i * batch_size + j]]) for j in range(batch_size) if i * batch_size + j < len(idxs)]
    resps = [ds['train'][idxs[i * batch_size + j]]['output'] + " </s>" for j in range(batch_size) if i * batch_size + j < len(idxs)]
    prompt_toks = tokenizer(ps, add_special_tokens=False)
    resp_toks = tokenizer(resps, add_special_tokens=False)

    all_toks = [p + r for p, r in zip(prompt_toks['input_ids'], resp_toks['input_ids'])]
    masked_toks = [[-100 for tok in p] + r for p, r in zip(prompt_toks['input_ids'], resp_toks['input_ids'])]
    all_toks = torch.nn.utils.rnn.pad_sequence([torch.tensor(t) for t in all_toks], batch_first=True, padding_value=tokenizer.eos_token_id)
    attn_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor([1] * len(t)) for t in all_toks], batch_first=True, padding_value=0)
    masked_toks = torch.nn.utils.rnn.pad_sequence([torch.tensor(t) for t in masked_toks], batch_first=True, padding_value=-100)

    def pad_truncate(x, round_len_to, max_length, pad_with):
        if x.shape[1] > max_length:
            x = x[:, :max_length]
        if x.shape[1] % round_len_to != 0:
            x = torch.nn.functional.pad(x, (0, round_len_to - (x.shape[1] % round_len_to)), value=pad_with)
        return x
    all_toks = pad_truncate(all_toks, 512, 2048, tokenizer.pad_token_id)
    attn_mask = pad_truncate(attn_mask, 512, 2048, 0)
    masked_toks = pad_truncate(masked_toks, 512, 2048, -100)

    with torch.cuda.amp.autocast():
        with torch.set_grad_enabled(True):
            input, target = all_toks[:, :-1], masked_toks[:, 1:]
            input, target = input.to(device), target.to(device)
            pred = model(input, attention_mask=attn_mask[:, :-1].to(device)).logits.float()
            loss = torch.nn.functional.cross_entropy(pred.transpose(1, 2), target)
            loss.backward()
            if (i % accum_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            n_toks = attn_mask.sum()
            dt = time() - t
            tok_per_sec = n_toks / dt
            t = time()
            wandb.log({"loss": loss.item(), "lr": lr, "tok_per_sec": tok_per_sec})
        
        with torch.set_grad_enabled(False):
            if (i % 100) == 0:
                eval_ps = ["<s> " + prompt(ds['train'][idx]) for idx in eval_idxs[:1]]
                eval_resps = [ds['train'][idx]['output'] + " </s>" for idx in eval_idxs[:1]]
                eval_prompt_toks = tokenizer(eval_ps, add_special_tokens=False)
                eval_resp_toks = tokenizer(eval_resps, add_special_tokens=False)

                gen_prompt_toks = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in eval_prompt_toks['input_ids']], batch_first=True, padding_value=tokenizer.eos_token_id)
                gen_attn_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(m) for m in eval_prompt_toks['attention_mask']], batch_first=True, padding_value=0)

                gen = model.generate(input_ids=gen_prompt_toks, attention_mask=gen_attn_mask, max_new_tokens=256, num_return_sequences=1, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
                gen = tokenizer.batch_decode(gen, skip_special_tokens=False)
                print(gen)


                table = wandb.Table(columns=["step", "prompt", "response", "ground_truth"])
                table.add_data(i, eval_ps[0], gen[0], eval_resps[0])
                wandb.log({"examples": table})

    gc.collect()


