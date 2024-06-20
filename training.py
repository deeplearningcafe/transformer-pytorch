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
from tqdm import tqdm

np.random.seed(46)
torch.manual_seed(46)

# dataset: WMT14 English German, huggingface 



def train(model, train_loader, val_loader, optim, scheduler, conf: DictConfig):
    current_step = 0
    current_epoch = 0
    logs = []
    train_losses = 0
    val_losses = 0
    grad_norms = 0
    # total_tokens = 0
    running_steps = 0
    if conf.train.epochs == 0:
        epochs = conf.train.steps / len(train_loader)
        pbar = tqdm(total=conf.train.steps)

    elif conf.train.steps == 0:
        steps = conf.train.epochs * len(train_loader)
        pbar = tqdm(total=conf.train.epochs)

    print("Start training!")
    print(f"Total Epochs {epochs} || Total steps {steps}")
    start_time = time.time()

    while current_epoch < epochs:
        for batch in train_loader:
            
            
            batch = {k: v.to(conf.train.device) for k, v in batch.items()}
            # print(batch["en"]["input_ids"].device, batch["en"]["input_ids"].shape)
            outputs, loss_train = model(batch["en"]["input_ids"], batch["de"]["input_ids"], batch["en"]["attention_mask"], batch["de"]["attention_mask"])
            
            
            train_losses += loss_train.item()
            # total_tokens += batch["en"]["input_ids"].shape[-1] + batch["de"]["input_ids"].shape[-1]
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
            grad_norms += norm.item()

            # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
            # nn.utils.clip_grad_value_(
            #     model.parameters(), clip_value=2.0)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optim.step()
            scheduler.step()


                
            if current_step % conf.train.val_steps == 0:
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(conf.train.device) for k, v in batch.items()}
                        outputs, loss_val = model(batch["en"]["input_ids"], batch["de"]["input_ids"], batch["en"]["attention_mask"], batch["de"]["attention_mask"])
                        val_losses += loss_val.item()

            if current_step % conf.train.log_steps == 0:
                # we want the loss per step so we divide by the num of steps that have been accumulated
                train_losses /= running_steps
                val_losses /= len(val_loader)
                grad_norms /= running_steps

                print(f"Step {current_step}  || Train Loss : {train_losses} || Validation Loss : {val_losses} || Step Time: {time.time()-start_time} || Learning rate: {scheduler.get_lr()[0]} || Norm: {grad_norms}" # || Trained Tokens: {total_tokens}"
                      )
                start_time = time.time()
                running_steps = 0

                log_epoch = {'step': current_step+1, 'train_loss': train_losses, 'val_loss': val_losses,
                                "gradient_norm": grad_norms, #"trained_tokens": total_tokens,
                                "learning_rate": scheduler.get_lr()[0]}
                logs.append(log_epoch)
                df = pd.DataFrame(logs)
                df.to_csv("log_output.csv", index=False)
                train_losses = 0
                val_losses = 0
                grad_norms = 0
            
            if current_step % conf.train.save_steps == 0 or current_step == steps-1:
                print("Saving")
                sd = model.state_dict()

                # we are using tied embeddings, so we don't need the lm_head
                del sd['lm_head.weight']
                torch.save(sd, conf.train.save_path + "transformer_" + 
                       str(current_step+1) + '.pth')
            
            current_step += 1
            current_epoch
            pbar.update(1)
            if current_step >= steps:
                break
        
        current_epoch += 1
            
            
    print("End Training")

def overfit_one_batch(model, batch, optim, scheduler, conf: DictConfig):
    batch = {k: v.to(conf.train.device) for k, v in batch.items()}
    losses = []
    grad_norms = []
    current_step = 0
    logs = []
    lrs = []
    pbar = tqdm(total=conf.overfit_one_batch.max_steps)
    print("Start overfitting in one batch!")
    while current_step < conf.overfit_one_batch.max_steps:
        outputs, loss_train = model(batch["en"]["input_ids"], batch["de"]["input_ids"], batch["en"]["attention_mask"], batch["de"]["attention_mask"])
          
        losses.append(loss_train.item())

        optim.zero_grad()
        loss_train.backward()

        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        grad_norms.append(norm.item())
        lrs.append(scheduler.get_lr()[0])

        # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
        # nn.utils.clip_grad_value_(
        #     model.parameters(), clip_value=2.0)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
        optim.step()
        scheduler.step()

        current_step += 1
        pbar.update(1)
        if current_step > 1 and abs(losses[-2]-losses[-1]) < conf.overfit_one_batch.tolerance:
            break


    log_epoch = {'losses': losses, "gradient_norm": grad_norms, "learning_rate": lrs}
    df = pd.DataFrame(log_epoch)
    df.to_csv("test.csv", index=False)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(conf: DictConfig):
    model, tokenizer, optim, scheduler = prepare_training(conf)
    train_loader, val_loader = prepare_data(tokenizer, conf)
    if not conf.overfit_one_batch.overfit:
        if os.path.isdir(conf.train.save_path) == False:
            os.makedirs(conf.train.save_path)
        train(model, train_loader, val_loader, optim, scheduler, conf)
        # train(model, train_loader, val_loader, optim, scheduler, steps=10, val_steps=10, log_steps=10, save_steps=10)
    else:
        batch = next(iter(train_loader))
        overfit_one_batch(model, batch, optim, scheduler, conf)

if __name__ == "__main__":
    main()