import torch
import numpy as np
from util.prepara_data import prepare_test_data
from util.prepare_model import prepare_test
import hydra
from omegaconf import DictConfig
np.random.seed(46)
torch.manual_seed(46)


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
    test_loader, eval_loader = prepare_test_data(tokenizer, conf)
    print(len(test_loader))

    test_losses = test(model, test_loader)
    val_losses = test(model, eval_loader)

    print(f"Total Val losses: {val_losses} | Val loss per sample : {val_losses/len(eval_loader)} ")
    print(f"Total Test losses: {test_losses} | Test loss per sample : {test_losses/len(test_loader)} ")

if __name__ == "__main__":
    main()