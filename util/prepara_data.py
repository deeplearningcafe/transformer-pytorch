import datasets
from torch.utils.data import default_collate
from torch.utils.data import DataLoader
import omegaconf
from abc import ABC, abstractmethod
from typing import Callable


class transformer_dataset(ABC):
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        
    def load_data(self):
        data = datasets.load_dataset(self.dataset_name)
        print("Dataset loaded!")
        train = data["train"]
        test = data["test"]
        val = data["validation"]

        return train, test, val
    
    @abstractmethod
    def collate_fn(batch, tokenizer, conf):
        return NotImplementedError
    
class WNT_dataset(transformer_dataset):

    def load_data(self):
        data = datasets.load_dataset(self.dataset_name, data_dir="de-en")
        print("Dataset loaded!")
        train = data["train"]
        test = data["test"]
        val = data["validation"]
        
        return train, test, val
    
    @staticmethod
    def collate_fn(batch, tokenizer, conf):
        batch = default_collate(batch)
        # we need to add special tokens
        inputs = tokenizer(batch["translation"]["en"],  max_length=conf.train.max_length, padding="longest", truncation=True, return_tensors="pt",
                        add_special_tokens=True, return_token_type_ids=False)
        outputs = tokenizer(batch["translation"]["de"],  max_length=conf.train.max_length, padding="longest", truncation=True, return_tensors="pt",
                            add_special_tokens=True, return_token_type_ids=False)

        return {"en": inputs, "de": outputs}

class multi30k_dataset(transformer_dataset):
    
    # we will pretokenize the dataset
    @staticmethod
    def collate_fn(batch, tokenizer, conf):
        batch = default_collate(batch)
        # we need to add special tokens
        inputs = tokenizer(batch["en"],  max_length=conf.train.max_length, padding="longest", truncation=True, return_tensors="pt",
                        add_special_tokens=True, return_token_type_ids=False)
        outputs = tokenizer(batch["de"],  max_length=conf.train.max_length, padding="longest", truncation=True, return_tensors="pt",
                            add_special_tokens=True, return_token_type_ids=False)

        return {"en": inputs, "de": outputs}

def create_loaders(train:datasets.Dataset, val:datasets.Dataset, tokenizer=None, conf=None, collate_fn:Callable=None):
    train_loader = DataLoader(train, batch_size=conf.train.train_batch, collate_fn=lambda batch: collate_fn(batch, tokenizer, conf), )
    val_loader = DataLoader(val, batch_size=conf.train.eval_batch, collate_fn=lambda batch: collate_fn(batch, tokenizer, conf), )
    # test_loader = DataLoader(test, batch_size=4, collate_fn=collate_fn)
    print(len(train_loader))
    print(len(val_loader))
    return train_loader, val_loader

def select_dataset(dataset_name:str):
    if dataset_name == "bentrevett/multi30k":
        dataset = multi30k_dataset(dataset_name)
    elif dataset_name == "wmt/wmt14":
        dataset = WNT_dataset(dataset_name)
    
    return dataset

def prepare_data(tokenizer=None, conf: omegaconf.DictConfig=None):
    dataset_name = conf.train.dataset_name
    dataset = select_dataset(dataset_name)
    
    train, test, val = dataset.load_data()
    train_loader, val_loader = create_loaders(train, val, tokenizer, conf, dataset.collate_fn)

    return train_loader, val_loader

def prepare_test_data(tokenizer=None, conf: omegaconf.DictConfig=None):
    dataset_name = conf.train.dataset_name
    dataset = select_dataset(dataset_name)
    
    train, test, eval = dataset.load_data()
    test_loader = DataLoader(test, batch_size=conf.train.eval_batch, collate_fn=lambda batch: dataset.collate_fn(batch, tokenizer, conf), )
    eval_loader = DataLoader(eval, batch_size=conf.train.eval_batch, collate_fn=lambda batch: dataset.collate_fn(batch, tokenizer, conf), )

    return test_loader, eval_loader