from torchtext import data, datasets

BOS = 2
EOS = 3

class DataLoader():

    def __init__(self, train_fn, valid_fn, batch_size = 64, device = -1, max_vocab = 99999999, max_length = 255, fix_length = None, use_bos = True, use_eos = True, shuffle = True):
        super(DataLoader, self).__init__()

        self.text = data.Field(sequential = True, 
                                use_vocab = True, 
                                batch_first = True, 
                                include_lengths = True, 
                                fix_length = fix_length, 
                                init_token = '<BOS>' if use_bos else None, 
                                eos_token = '<EOS>' if use_eos else None
                                )

        train = LanguageModelDataset(path = train_fn, 
                                        fields = [('text', self.text)], 
                                        max_length = max_length
                                        )
        valid = LanguageModelDataset(path = valid_fn, 
                                        fields = [('text', self.text)], 
                                        max_length = max_length
                                        )

        self.train_iter = data.BucketIterator(train, 
                                                batch_size = batch_size, 
                                                device = device, 
                                                shuffle = shuffle, 
                                                sort_key=lambda x: -len(x.text), 
                                                sort_within_batch = True
                                                )
        self.valid_iter = data.BucketIterator(valid, 
                                                batch_size = batch_size, 
                                                device = device, 
                                                shuffle = False, 
                                                sort_key=lambda x: -len(x.text), 
                                                sort_within_batch = True
                                                )

        self.text.build_vocab(train, max_size = max_vocab)

class TextClassificationDataLoader():

    def __init__(self, train_fn, valid_fn, tokenizer, batch_size = 64, device = -1, max_vocab = 9999999, fix_length = None, use_eos = False, shuffle = True):
        super(TextClassificationDataLoader, self).__init__()

        self.label = data.Field(sequential = False, use_vocab = False)
        self.text = data.Field(tokenize = tokenizer, use_vocab = True, batch_first = True, include_lengths = True, fix_length = fix_length, eos_token = '<EOS>' if use_eos else None)

        train, valid = data.TabularDataset.splits(path = '', train = train_fn, validation = valid_fn, format = 'tsv', fields = [('label', self.label), ('text', self.text)])

        self.train_iter, self.valid_iter = data.BucketIterator.splits((train, valid), batch_size = batch_size, device = device, shuffle = shuffle)
        
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size = max_vocab)

class LanguageModelDataset(data.Dataset):

    def __init__(self, path, fields, max_length=None, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('text', fields[0])]

        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if max_length and max_length < len(line.split()):
                    continue
                if line != '':
                    examples.append(data.Example.fromlist(
                        [line], fields))

        super(LanguageModelDataset, self).__init__(examples, fields, **kwargs)


if __name__ == '__main__':
    import sys
    loader = DataLoader(sys.argv[1], sys.argv[2])

    for batch_index, batch in enumerate(loader.train_iter):
        print(batch.text)

        if batch_index > 1:
            break
