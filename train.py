import gc
import math

import wandb
import torch
from torch import nn
import matplotlib.pyplot as plt

from einops import rearrange
from torch.utils.cpp_extension import load, load_inline
from kac import KacRandomWalk_, kac_random_walk_

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

batch_size = 8

class KacLayer(nn.Module):
    def __init__(self, layer, dim, seed=2024, n_steps=None, adapter_dtype=torch.float32):
        super().__init__()
        inner_w = next(layer.parameters())
        self.vec = nn.Parameter(torch.randn(dim, device=inner_w.device, dtype=adapter_dtype) * 1/math.sqrt(dim))
        layer.requires_grad_(False)
        self.layer = layer
        self.dim = dim
        self.seed = seed
        self.n_steps = n_steps or math.ceil(math.log2(dim)) * dim

    def forward(self, x):
        y = self.layer(x)
        x_2d = x.view(-1, self.dim).to(self.vec.dtype)
        y_p = kac_random_walk_(x_2d.clone(), self.seed * 2, self.n_steps)
        y_p = self.vec * y_p.to(self.vec.dtype)
        
        y_p = kac_random_walk_(y_p.clone(), self.seed * 2 + 1, self.n_steps)
        y_p = y_p.view_as(y).to(y.dtype)
        return y + y_p

model = transformers.AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", load_in_8bit=True, torch_dtype=torch.bfloat16)
print(model)

def replace_layer(layer_idx, m, dim, seed=2024):
    if hasattr(m, "__wrapped__"):
        return
    if isinstance(m, LlamaAttention):
        m.q_proj = KacLayer(m.q_proj, dim, layer_idx * 4 + 1 + seed)
        m.k_proj = KacLayer(m.k_proj, dim, layer_idx * 4 + 2 + seed)
        m.v_proj = KacLayer(m.v_proj, dim, layer_idx * 4 + 3 + seed)
        m.o_proj = KacLayer(m.o_proj, dim, layer_idx * 4 + 4 + seed)
        m.__wrapped__ = True

for i, l in enumerate(model.model.layers):
    replace_layer(i, l.self_attn, seed=2024, dim=4096)

for name, module in model.named_modules():
    if "norm" in name:
        module.to(torch.float32)
    if "lm_head" in name:
        if hasattr(module, "weight"):
            module.to(torch.bfloat16)


def freeze_linear(layer):
    if not isinstance(layer, KacLayer):
        for p in layer.parameters():
            p.requires_grad_(False)

accum_steps = 4
warmup_steps = 10
total_steps = 100

def cosine_lr_with_warmup(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return 0.5 * (1 + math.cos((step - warmup_steps) / (total_steps - warmup_steps) * math.pi))

target_lr = 1e-4

wandb.init(project="kac-llama", entity="uwu1", name="kac-llama-7b")


model.model.apply(freeze_linear)
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
print(model)
for i in tqdm(range(total_steps)):
    lr = target_lr * cosine_lr_with_warmup(i, warmup_steps, total_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


    ps = [prompt(ds['train'][i * batch_size + j]) for j in range(batch_size)]
    resps = [ds['train'][i * batch_size + j]['output'] for j in range(batch_size)]
    prompt_toks = tokenizer(ps)
    resp_toks = tokenizer(resps)

    all_toks = [p + r for p, r in zip(prompt_toks['input_ids'], resp_toks['input_ids'])]
    masked_toks = [[-100 for tok in p] + r for p, r in zip(prompt_toks['input_ids'], resp_toks['input_ids'])]
    all_toks = torch.nn.utils.rnn.pad_sequence([torch.tensor(t) for t in all_toks], batch_first=True, padding_value=tokenizer.eos_token_id)
    attn_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor([1] * len(t)) for t in all_toks], batch_first=True, padding_value=0)
    masked_toks = torch.nn.utils.rnn.pad_sequence([torch.tensor(t) for t in masked_toks], batch_first=True, padding_value=-100)
    with torch.cuda.amp.autocast():
        with torch.set_grad_enabled(True):
            input, target = all_toks[:, :-1], masked_toks[:, 1:]
            input, target = input.cuda(), target.cuda()
            pred = model(input, attention_mask=attn_mask[:, :-1].cuda()).logits.float()
            loss = torch.nn.functional.cross_entropy(pred.transpose(1, 2), target)
            loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            print((masked_toks != -100).sum())
            print(loss)
            wandb.log({"loss": loss.item(), "lr": lr})
    gc.collect()


