from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import datasets
import numpy as np
np.random.seed(1234)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "[START]", "[END]", "[PAD]"], vocab_size=37000)

data = datasets.load_dataset("wmt/wmt14", data_dir="de-en")

print(data['train'])

def dataset_iter():
    for item in data['train']:
        num = np.random.randint(0, 2)
        if num == 0:
            yield item['translation']["en"]
        else:
            yield item['translation']["de"]

tokenizer.train_from_iterator(dataset_iter(), trainer)
tokenizer.save('data/tokenizer.json')
