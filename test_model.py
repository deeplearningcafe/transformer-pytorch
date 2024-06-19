import torch
import numpy as np
from util.prepara_data import prepare_test_data
from util.prepare_model import prepare_test
import hydra
from omegaconf import DictConfig
np.random.seed(46)
torch.manual_seed(46)


# data = datasets.load_dataset("wmt/wmt14", data_dir="de-en", )
# print("Dataset loaded!")
# test = data["test"]
# val = data["validation"]


# conf = {"hidden_dim": 512, "vocabulary_size": 12000, "num_heads": 8, "intermediate_dim": 2048, "eps": 1e-06, "num_layers": 6, "dropout": 0.0,
#             "label_smoothing": 0.1, "warmup_steps": 4000, "max_length": 128}
# device = "cuda"

# tokenizer = PreTrainedTokenizerFast(tokenizer_file=r"C:\Users\Victor\Deep Learning\papers_implementation\transformers\data\tokenizer.json")
# tokenizer.pad_token = "[PAD]"
# tokenizer.eos_token = "[PAD]"
# conf["vocabulary_size"] = tokenizer.vocab_size + 1
# model = transformer(conf).to(device)
# net_weights = torch.load(r'weights\transformer_200001.pth',
#                          map_location={'cuda:0': 'cpu'})

# model.load_state_dict(net_weights)
# model.eval()


# # we will pretokenize the dataset
# def collate_fn(batch):
#     batch = default_collate(batch)

#     inputs = tokenizer(batch["translation"]["en"],  max_length=conf["max_length"], padding="longest", truncation=True, return_tensors="pt")
#     outputs = tokenizer(batch["translation"]["de"],  max_length=conf["max_length"], padding="longest", truncation=True, return_tensors="pt")

#     return {"en": inputs, "de": outputs}

# test_loader = DataLoader(test, batch_size=16, collate_fn=collate_fn, )
# val_loader = DataLoader(val, batch_size=4, collate_fn=collate_fn, )

# print(len(test_loader))
# print(len(val_loader))

def test(model, test_loader, conf):
    
    test_losses = 0
    for batch in test_loader:
        
        batch = {k: v.to(conf.train.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs, loss = model(batch["en"]["input_ids"], batch["de"]["input_ids"], batch["en"]["attention_mask"], batch["de"]["attention_mask"])

        test_losses += loss.item()
        
    return test_losses


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig):

    model, tokenizer = prepare_test(conf)
    test_loader = prepare_test_data("bentrevett/multi30k", tokenizer, conf)
    print(len(test_loader))

    test_losses = test(model, test_loader)

    # val_losses = test(model, val_loader)
    # print(f"Total Val losses: {val_losses} | Val loss per sample : {val_losses/len(val_loader)} ")
    print(f"Total Test losses: {test_losses} | Test loss per sample : {test_losses/len(test_loader)} ")

if __name__ == "__main__":
    main()