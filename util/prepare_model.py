import torch.nn.init as init
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from transformer_implementation import transformer
from tokenizers.processors import TemplateProcessing
import torch
import math

def weights_init(m):
    if isinstance(m, nn.Linear):
        # init.xavier_normal_(m.weight.data)
        # 512 is the hidden_dim
        init.normal_(m.weight.data, mean=0, std=torch.sqrt(2/(5*512)))
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)

def apply_weights_init(model, conf):
    std = math.sqrt(2/(5*conf.transformer.hidden_dim))
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'gamma' in name.split('.') or 'beta' in name.split('.'):
                continue
            if 'bias' in name.split('.'):
                nn.init.constant_(param.data, 0.0)
            elif 'output_projection' in name.split('.'):
                init.normal_(param.data, mean=0, std=std/math.sqrt(2*conf.transformer.num_layers))
            else:
                init.normal_(param.data, mean=0, std=std)

class transformer_scheduler(LRScheduler):
    def __init__(self, optimizer, hidden_dim, warmup_steps, last_epoch=-1, verbose="deprecated") -> None:
        self.hidden_dim = hidden_dim
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr_rate = self.hidden_dim**(-0.5) * min(self._step_count**(-0.5), self._step_count*self.warmup_steps**(-1.5))
        return [lr_rate for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        lr_rate = self.hidden_dim**(-0.5) * min(self._step_count**(-0.5), self._step_count*self.warmup_steps**(-1.5))
        return [lr_rate for base_lr in self.base_lrs]
    
def create_model(conf, apply_init:bool=True):
    # vocab 37000 tokens, batch 25000 tokens per batch
    # optimizer: adam, beta1=0.9, beta2=0.98, e=10**-9. 
    # Learning rate: lrate= hidden_dim**-0.5 * min(step_num**-0.5, step_num * warmup_steps**-1.5) with warmup_steps=4000
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=r"".join(conf.tokenizer.tokenizer_path), bos_token="<bos>", eos_token="<eos>",
                pad_token="<pad>", unk_token="<unk>")
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor =TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id), 
            (f"{eos}", tokenizer.eos_token_id)
        ],
    )

    # conf.transformer.vocabulary_size = tokenizer.vocab_size + 1
    model = transformer(conf)
    if apply_init:
        # model.apply(weights_init)
        apply_weights_init(model, conf)
        model.set_tied_embeddings()
        
    model = model.to(conf.train.device)
    return model, tokenizer

def create_optim(conf, model):
    if conf.train.use_bitsandbytes == True:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optim = bnb.optim.AdamW8bit(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    else:
        optim = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = transformer_scheduler(optim, conf.transformer.hidden_dim, conf.train.warmup_steps)
    return optim, scheduler

def prepare_training(conf):
    model, tokenizer = create_model(conf)
    model.train()
    print("Model loaded")
    optim, scheduler = create_optim(conf, model)
    print("Optim loaded")

    return model, tokenizer, optim, scheduler

def prepare_test(conf):
    model, tokenizer = create_model(conf, apply_init=False)
    net_weights = torch.load(r''.join(conf.inference.checkpoint),
                         map_location={'cuda:0': 'cpu'})

    model.load_state_dict(net_weights)
    # tied embeddings after loading the weights
    model.set_tied_embeddings()
    model.eval()
    return model, tokenizer

def get_update_ratio(model, layers:dict[str, torch.tensor], diffs: dict[str, list]):
    # we have the layers dict that stores the weights and the diffs dict that stores a list with the diff in each step
    # dict are not in place so we need to create a new one
    tmp_dict = {}

    # the idea is to 
    for key, layer in layers.items():
        for name, param in model.named_parameters():
            # print(key)
            if key in name.split('.'):
                with torch.no_grad():
                    change = param-layer
                    # print(change.std()/param.std())
                    diffs[key].append((change.std()/param.std()).log10().item())
                    tmp_dict[key] = param.detach().cpu().clone()
    return tmp_dict, diffs