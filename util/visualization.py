import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_logs(logs_path:str):
    df = pd.read_csv(r"".join(logs_path))
    print(f"Max norm: {df['max_norm'].max()}")
    print(f"Min train loss: {df['train_loss'].min()}")
    print(f"Min val loss: {df[df['val_loss']!=0.0]['val_loss'].min()}")

    x = np.arange(len(df["train_loss"]))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, df["train_loss"],  label='train_loss')
    ax.scatter(x, df["val_loss"], label='val_loss')
    ax.plot(x, df["mean_norm"],  label='mean_norm')
    ax.plot(x, df["max_norm"],  label='max_norm')

    ax.plot(x, df["learning_rate"]*1000, label='lr')

    ax.set(xlabel='steps', ylabel='loss and gradients',)
    ax.grid()
    ax.set_ylim(ymin=0, ymax=12)
    ax.legend()
    plt.show()

def plot_activation_layer(activations_array: list[torch.tensor]):
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i, layer in enumerate(activations_array): # note: exclude the output layer
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, "ReLU", layer.mean(),layer.std(), (layer.abs() == 0.00).float().mean()*100))
        hy, hx = torch.histogram(layer, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ')
    plt.legend(legends)
    plt.title('activation distribution')
    plt.show()

def plot_gradients(layers_list:list[str], model:torch.nn.Module):
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []

    for name, param in model.named_parameters():
        t = param.grad
        if name in layers_list:
            print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(param.shape), t.mean(), t.std(), t.std() / param.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{tuple(param.shape)} {name}')
    plt.legend(legends)
    plt.title('weights gradient distribution')
    plt.show()

def plot_attention_map(attentions_array: list[torch.tensor]):
    attentions_numpy_0 = attentions_array[0][0][0].detach().numpy()
    attentions_numpy_5 = attentions_array[-1][0][0].detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(attentions_numpy_0)
    axs[0].set_title("Attention map first layer")
    axs[1].imshow(attentions_numpy_5)
    axs[1].set_title("Attention map last layer")

    plt.show()
