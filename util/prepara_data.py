import datasets
from torch.utils.data import default_collate
from torch.utils.data import DataLoader
import omegaconf

def load_data(conf: omegaconf.DictConfig):
    data = datasets.load_dataset(conf.train.dataset_name)
    print("Dataset loaded!")
    train = data["train"]
    test = data["test"]
    val = data["validation"]

    return train, test, val

# we will pretokenize the dataset
def collate_fn(batch, tokenizer, conf):
    batch = default_collate(batch)
    # we need to add special tokens
    inputs = tokenizer(batch["en"],  max_length=conf.train.max_length, padding="longest", truncation=True, return_tensors="pt",
                       add_special_tokens=True, return_token_type_ids=False)
    outputs = tokenizer(batch["de"],  max_length=conf.train.max_length, padding="longest", truncation=True, return_tensors="pt",
                        add_special_tokens=True, return_token_type_ids=False)

    return {"en": inputs, "de": outputs}

def create_loaders(train:datasets.Dataset, val:datasets.Dataset, tokenizer=None, conf=None):
    train_loader = DataLoader(train, batch_size=conf.train.train_batch, collate_fn=lambda batch: collate_fn(batch, tokenizer, conf), )
    val_loader = DataLoader(val, batch_size=conf.train.eval_batch, collate_fn=lambda batch: collate_fn(batch, tokenizer, conf), )
    # test_loader = DataLoader(test, batch_size=4, collate_fn=collate_fn)
    print(len(train_loader))
    print(len(val_loader))
    return train_loader, val_loader

def prepare_data(tokenizer=None, conf: omegaconf.DictConfig=None):
    train, test, val = load_data(conf)
    train_loader, val_loader = create_loaders(train, val, tokenizer, conf)

    return train_loader, val_loader

def prepare_test_data(tokenizer=None, conf: omegaconf.DictConfig=None):
    test = datasets.load_dataset(conf.train.dataset_name, split="test")
    test_loader = DataLoader(test, batch_size=16, collate_fn=lambda batch: collate_fn(batch, tokenizer, conf), )
    eval = datasets.load_dataset(conf.train.dataset_name, split="validation")
    eval_loader = DataLoader(eval, batch_size=16, collate_fn=lambda batch: collate_fn(batch, tokenizer, conf), )

    return test_loader, eval_loader