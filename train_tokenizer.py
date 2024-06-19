from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import datasets
import numpy as np
import os
import omegaconf
import hydra
np.random.seed(1234)

def load_tokenizer_data(conf: omegaconf.DictConfig):
    tokenizer = Tokenizer(BPE(unk_token=conf.tokenizer.unk_token))
    tokenizer.pre_tokenizer = Whitespace()
    conf_dict = omegaconf.OmegaConf.to_object(conf)
    trainer = BpeTrainer(special_tokens=conf_dict["tokenizer"]["special_tokens"], vocab_size=conf.tokenizer.vocabulary_size)

    data = datasets.load_dataset(conf.train.dataset_name, split="train")

    print(data)
    return tokenizer, trainer, data

def dataset_iter(data:datasets.Dataset, sampling:bool=False):
    for item in data:
        if sampling:
            num = np.random.randint(0, 2)
            if num == 0:
                yield item["en"]
            else:
                yield item["de"]
        else:
            yield item["en"] + " " + item["de"]

def train(tokenizer: Tokenizer, trainer: BpeTrainer, conf:omegaconf.DictConfig, data:datasets.Dataset):
    tokenizer.train_from_iterator(dataset_iter(data, False), trainer)
    save_dir = conf.tokenizer.save_path
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    tokenizer.save(f'{save_dir}/tokenizer.json')

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(conf: omegaconf.DictConfig):

    tokenizer, trainer, data = load_tokenizer_data(conf)
    train(tokenizer, trainer, conf, data)
    

if __name__ == "__main__":
    main()