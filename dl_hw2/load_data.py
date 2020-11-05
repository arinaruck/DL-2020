from torchtext import data, datasets
from torchtext.data import Iterator, BucketIterator
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_datasets(train, val, test):
    SRC = data.Field(
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        batch_first=True,
    )

    TRG = data.Field(
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        batch_first=True,
    )   
    train_dataset = datasets.TranslationDataset(
        path=train, exts=('.de', '.en'),
        fields=(SRC, TRG))

    val_dataset = datasets.TranslationDataset(
        path=val, exts=('.de', '.en'),
        fields=(SRC, TRG))

    TRG.build_vocab(train_dataset, min_freq=2)
    SRC.build_vocab(train_dataset, min_freq=2)

    print(len(SRC.vocab.itos), len(TRG.vocab.itos))

    train_iter, val_iter = BucketIterator.splits((train_dataset, val_dataset), 
        batch_sizes=(64, 64),
        device=device, 
        sort_key=lambda x: len(x.src), 
        sort_within_batch=True,
        repeat=False
    )

    return train_iter, val_iter, SRC, TRG

