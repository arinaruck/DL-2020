from torchtext import data, datasets
from torchtext.data import Iterator, BucketIterator
import torch from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decoder_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.uint8), diagonal=1)
    return subsequent_mask == 0  


class BucketIteratorWrapper(DataLoader):
    __initialized = False

    def __init__(self, iterator: data.BucketIterator):
        self.batch_size = iterator.batch_size
        self.num_workers = 1
        self.collate_fn = None
        self.pin_memory = True
        self.drop_last = False
        self.timeout = 0
        self.worker_init_fn = None
        self.sampler = iterator
        self.batch_sampler = iterator
        self.__initialized = True

    def __iter__(self):
        return map(
            lambda batch: Batch(batch.src, batch.trg, pad=1),
            self.batch_sampler.__iter__()
        )

    def __len__(self):
        return len(self.batch_sampler)


class Batch():
    def __init__(self, src, trg=None, pad=0):
        self.trg = trg[:, :-1]
        self.pred = trg[:, 1:]
        self.trg_mask = self.make_mask(self.trg, pad)
        self.src = src
        self.src_mask = (src != pad).unsqueeze(1)
    
    @staticmethod
    def make_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(1)
        tgt_mask = tgt_mask & decoder_mask(tgt.shape[-1]).detach().to(device)
        return tgt_mask


def make_datasets(config, train, val):
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

    TRG.build_vocab(train_dataset, min_freq=3)
    SRC.build_vocab(train_dataset, min_freq=3)

    print(len(SRC.vocab.itos), len(TRG.vocab.itos))

    train_iter, val_iter = data.Iterator.splits((train_dataset, val_dataset), 
                                                           batch_sizes=(config['batch_size'], config['batch_size']), 
                                      sort_key=lambda x: len(x.src) + len(x.trg),
                                      shuffle=True,
                                      device=DEVICE,
                                      sort_within_batch=False)
                                     
    train_iter = BucketIteratorWrapper(train_iter)
    val_iter = BucketIteratorWrapper(val_iter)

    return train_iter, val_iter, SRC, TRG

