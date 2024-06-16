import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import datasets
from transformer_implementation import transformer
from torch.utils.data import default_collate
from transformers import PreTrainedTokenizerFast
import bitsandbytes as bnb
import time
import torch.nn.init as init
import torch.nn as nn
import numpy as np
import pandas as pd

np.random.seed(46)
torch.manual_seed(46)

# dataset: WMT14 English German, huggingface 
data = datasets.load_dataset("wmt/wmt14", data_dir="de-en")
print("Dataset loaded!")
train = data["train"]
test = data["test"]
val = data["validation"]
conf = {"hidden_dim": 512, "vocabulary_size": 12000, "num_heads": 8, "intermediate_dim": 2048, "eps": 1e-06, "num_layers": 6, "dropout": 0.1,
            "label_smoothing": 0.1, "warmup_steps": 4000, "max_length": 128}
device = "cuda"


def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)

# vocab 37000 tokens, batch 25000 tokens per batch
# optimizer: adam, beta1=0.9, beta2=0.98, e=10**-9. 
# Learning rate: lrate= hidden_dim**-0.5 * min(step_num**-0.5, step_num * warmup_steps**-1.5) with warmup_steps=4000
tokenizer = PreTrainedTokenizerFast(tokenizer_file=r"C:\Users\Victor\Deep Learning\papers_implementation\transformers\data\tokenizer.json")
tokenizer.pad_token = "[PAD]"
tokenizer.eos_token = "[PAD]"
conf["vocabulary_size"] = tokenizer.vocab_size + 1
model = transformer(conf)
model.apply(weights_init)
model = model.to(device)
model.train()
# print(model)
# for param in model.parameters():
#     print(param)
# print(model.device)

# exit()
print("Model loaded!")

# optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
optim = bnb.optim.AdamW8bit(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

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

scheduler = transformer_scheduler(optim, conf["hidden_dim"], conf["warmup_steps"])

# we will pretokenize the dataset
def collate_fn(batch):
    batch = default_collate(batch)

    # we need to add special tokens
    inputs = tokenizer(batch["translation"]["en"],  max_length=conf["max_length"], padding="longest", truncation=True, return_tensors="pt",
                       add_special_tokens=True)
    outputs = tokenizer(batch["translation"]["de"],  max_length=conf["max_length"], padding="longest", truncation=True, return_tensors="pt")

    return {"en": inputs, "de": outputs}

train_loader = DataLoader(train, batch_size=14, collate_fn=collate_fn, )
val_loader = DataLoader(val, batch_size=4, collate_fn=collate_fn, )
# test_loader = DataLoader(test, batch_size=4, collate_fn=collate_fn)
print(len(train_loader))
print(len(val_loader))


def train(model, train_loader, val_loader, optim, scheduler, steps:int, val_steps:int, log_steps:int, save_steps:int):
    current_step = 0
    logs = []
    train_losses = 0
    val_losses = 0
    print("Start training!")
    while current_step < steps:
        start_time = time.time()
        for batch in train_loader:
            
            
            batch = {k: v.to(device) for k, v in batch.items()}
            # print(batch["en"]["input_ids"].device, batch["en"]["input_ids"].shape)
            outputs, loss_train = model(batch["en"]["input_ids"], batch["de"]["input_ids"], batch["en"]["attention_mask"], batch["de"]["attention_mask"])
            
            
            train_losses += loss_train.item()
            
            if current_step % log_steps == 0:
                print(f"Step {current_step}  || Loss : {train_losses} || Step Time: {time.time()-start_time} || Learning rate: {scheduler.get_lr()}")
                start_time = time.time()
            
            optim.zero_grad()
            loss_train.backward()
            
            # print last layer gradients
            for p in model.decoder.layers[0].msha.parameters():
                print(p)
            
            # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
            # nn.utils.clip_grad_value_(
            #     model.parameters(), clip_value=2.0)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, norm_type=2)
            optim.step()
            scheduler.step()
        
            if current_step % val_steps == 0:
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs, loss_val = model(batch["en"]["input_ids"], batch["de"]["input_ids"], batch["en"]["attention_mask"], batch["de"]["attention_mask"])
                        val_losses += loss_val.item()
                        
                    print(f"Validation Step {current_step}: {torch.mean(loss_val)}")
                                # ログを保存
                    log_epoch = {'step': current_step+1,
                                'train_loss': train_losses, 'val_loss': val_losses}
                    logs.append(log_epoch)
                    df = pd.DataFrame(logs)
                    df.to_csv("log_output.csv")
                    train_losses = 0
                    val_losses = 0
            
            if current_step % save_steps == 0 or current_step == steps:
                print("Saving")
                torch.save(model.state_dict(), 'weights/transformer_' +
                       str(current_step+1) + '.pth')
            
            current_step += 1
            if current_step >= steps:
                break
        

            
            
    print("End Training")
train(model, train_loader, val_loader, optim, scheduler, steps=1000000, val_steps=1000, log_steps=1000, save_steps=1000)
# train(model, train_loader, val_loader, optim, scheduler, steps=10, val_steps=10, log_steps=10, save_steps=10)
