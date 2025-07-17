import json
import re
from collections import Counter
from configs.config import DataConfig


class Tokenizer(object):
    def __init__(self, config = DataConfig()):
        self.annotation_path = config.annotation_path
        self.threshold = config.threshold
        self.dataset_name = config.dataset_name
        self.clean_report = self.clean_report_iu_xray
        self.ann = json.loads(open(self.annotation_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

        self.pad_token_id = self.token2idx['<pad>']
        self.bos_token_id = self.token2idx['<bos>']
        self.eos_token_id = self.token2idx['<eos>']

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold]
        vocab.sort()

        # Add special tokens at the beginning
        vocab = ['<pad>', '<bos>', '<eos>', '<unk>'] + vocab + ['<image>']
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for token, idx in token2idx.items()}

        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        return ' . '.join(tokens) + ' .'

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = [self.bos_token_id] + [self.get_id_by_token(t) for t in tokens] + [self.eos_token_id]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx.item()]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        return [self.decode(ids) for ids in ids_batch]