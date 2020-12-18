# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
from typing import Dict, List, Union


class Vocab(object):

    def __init__(self, data: List[str]):
        super(Vocab, self).__init__()
        self.token2id: Dict[str, int] = dict()
        self.id2token = []

        for gold in data:
            labels = gold.split(' ')
            for label in labels:
                if label not in self.token2id:
                    self.token2id[label] = len(self.token2id)
                    self.id2token.append(label)

    def __len__(self):
        return len(self.id2token)

    def encode(self, tokens: Union[str, List[str], List[List[str]]]) -> Union[int, List[int], List[List[int]]]:
        """encode str to idx.

        Notes:
            a word -> str
            a sentence -> List[str], you must not assign `str` to a sentence!!!
            a batch sentences -> List[List[str]]

        Notes:
            labels must exist in vocabs, so do not get defult [UNK] id.
            Otherwise, you should fix code self.token2id.get(tokens, [UNK])
        """
        if isinstance(tokens, str):
            # labels must exist in vocabs, so do not get defult [UNK] id.
            # Otherwise, you should fix code self.token2id.get(tokens, [UNK])
            return self.token2id.get(tokens)
        else:
            return [self.encode(token) for token in tokens]

    def decode(self, idxs: Union[int, List[int], List[List[int]]]) -> Union[str, List[str], List[List[str]]]:
        """decode idx to str.

        inverse of self.encode()

        Notes:
            a word id -> id
            a sentence ids -> List[int], you must not assign `int` to a sentence!!!
            a batch sentences ids -> List[List[int]]

        Notes:
            labels must exist in vocabs, so do not get defult [UNK] id.
        """
        if isinstance(idxs, int):
            return self.id2token[idxs]
        else:
            return [self.decode(idx) for idx in idxs]
