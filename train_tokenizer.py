from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import datasets
import numpy as np
import os
np.random.seed(1234)

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"], vocab_size=37000)

data = datasets.load_dataset("bentrevett/multi30k", split="train")

print(data)

def dataset_iter(sampling:bool=False):
    for item in data:
        if sampling:
            num = np.random.randint(0, 2)
            if num == 0:
                yield item["en"]
            else:
                yield item["de"]
        else:
            yield item["en"] + " " + item["de"]

tokenizer.train_from_iterator(dataset_iter(False), trainer)
save_dir = "data"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
tokenizer.save(f'{save_dir}/tokenizer.json')
