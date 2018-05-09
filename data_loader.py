from torchtext import data, datasets

BOS = 2
EOS = 3

class DataLoader():

    def __init__(self, train_fn, valid_fn, batch_size = 64, device = -1, max_vocab = 99999999, max_length = 255, fix_length = None, use_bos = True, use_eos = True, shuffle = True):
        super(DataLoader, self).__init__()

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

