import torch
import time
import torch.nn as nn
import numpy as np
import pandas as pd
from util.prepara_data import prepare_data
from util.prepare_model import prepare_training
import hydra
from omegaconf import DictConfig
import os

np.random.seed(46)
torch.manual_seed(46)

# dataset: WMT14 English German, huggingface 



def train(model, train_loader, val_loader, optim, scheduler, conf: DictConfig):
    current_step = 0
    logs = []
    train_losses = 0
    # val_losses = 0
    total_tokens = 0
    running_steps = 0
    print("Start training!")
    while current_step < conf.train.steps:
        start_time = time.time()
        for batch in train_loader:
            
            
            batch = {k: v.to(conf.train.device) for k, v in batch.items()}
            # print(batch["en"]["input_ids"].device, batch["en"]["input_ids"].shape)
            outputs, loss_train = model(batch["en"]["input_ids"], batch["de"]["input_ids"], batch["en"]["attention_mask"], batch["de"]["attention_mask"])
            
            
            train_losses += loss_train.item()
            total_tokens += batch["en"]["input_ids"].shape[-1] + batch["de"]["input_ids"].shape[-1]
            running_steps += 1

            optim.zero_grad()
            loss_train.backward()
            
            # print last layer gradients
            # current_grad = []
            # for p in model.decoder.layers[0].msha.parameters():
            #     if p.grad is not None:
            #         current_grad.append(p.grad.detach().flatten())
            # norm = torch.cat(current_grad).norm()
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            norm = torch.cat(grads).norm()


            # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
            # nn.utils.clip_grad_value_(
            #     model.parameters(), clip_value=2.0)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optim.step()
            scheduler.step()

            if current_step % conf.train.log_steps == 0:
                # we want the loss per step so we divide by the num of steps that have been accumulated
                train_losses /= running_steps
                print(f"Step {current_step}  || Train Loss : {train_losses} || Step Time: {time.time()-start_time} || Learning rate: {scheduler.get_lr()[0]} || Norm: {norm} || Trained Tokens: {total_tokens}")
                start_time = time.time()
                running_steps = 0
                
            if current_step % conf.train.val_steps == 0:
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(conf.train.device) for k, v in batch.items()}
                        outputs, loss_val = model(batch["en"]["input_ids"], batch["de"]["input_ids"], batch["en"]["attention_mask"], batch["de"]["attention_mask"])
                        # val_losses += loss_val.item()
                    
                    # val_losses /= conf.train.val_steps
                    print(f"Validation Step {current_step}: {torch.mean(loss_val)}")
                                # ログを保存
                    log_epoch = {'step': current_step+1, 'train_loss': train_losses, 'val_loss': torch.mean(loss_val),
                                  "gradient_norm": norm, "trained_tokens": total_tokens,
                                  "learning_rate": scheduler.get_lr()[0]}
                    logs.append(log_epoch)
                    df = pd.DataFrame(logs)
                    df.to_csv("log_output.csv")
                    train_losses = 0
                    # val_losses = 0
            
            if current_step % conf.train.save_steps == 0 or current_step == conf.train.steps-1:
                print("Saving")
                torch.save(model.state_dict(), conf.train.save_path + "transformer_" + 
                       str(current_step+1) + '.pth')
            
            current_step += 1
            if current_step >= conf.train.steps:
                break
        

            
            
    print("End Training")

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(conf: DictConfig):

    model, tokenizer, optim, scheduler = prepare_training(conf)
    train_loader, val_loader = prepare_data(tokenizer, conf)
    if os.path.isdir(conf.train.save_path) == False:
        os.makedirs(conf.train.save_path)
    train(model, train_loader, val_loader, optim, scheduler, conf)
    # train(model, train_loader, val_loader, optim, scheduler, steps=10, val_steps=10, log_steps=10, save_steps=10)

if __name__ == "__main__":
    main()