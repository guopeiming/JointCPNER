# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
from typing import List, Set, Tuple


class NERFScore(object):
    def __init__(self, true_positive: int, false_positive: int, false_negetive: int):
        super(NERFScore, self).__init__()

        recall = 0.
        if true_positive+false_negetive != 0:
            recall = true_positive/(true_positive+false_negetive)

        precision = 0.
        if true_positive+false_positive != 0:
            precision = true_positive/(true_positive+false_positive)

        fscore = 0.
        if recall+precision != 0:
            fscore = (2*recall*precision)/(recall+precision)

        self.recall = recall * 100
        self.precision = precision * 100
        self.fscore = fscore * 100

    def __str__(self):
        return 'P: %.02f, R: %.02f, F1: %.02f' % (self.precision, self.recall, self.fscore)


def _bio_tag_to_spans(tags: List[str], ignore_labels=None) -> Set[Tuple[str, Tuple[int, int]]]:
    """给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O'].
        返回[('singer', (1, 4))] (左闭右开区间)
    Args:
        tags: List[str],
        ignore_labels: List[str], 在该list中的label将被忽略

    Returns:
        Set[Tuple[str, Tuple[int, int]]]. {(label，(start, end))}
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i':
            if prev_bio_tag in ('b', 'i') and len(spans) > 0 and label == spans[-1][0]:
                spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            print('error %s tag in BIO tags' % bio_tag)
        prev_bio_tag = bio_tag
    return {(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels}


def cal_performance(tags_pred: List[List[str]], tags_gold: List[List[str]], ignore_labels=None) -> NERFScore:
    """计算基于span的F1、P、R.
    Args:
        tags_pred: 裁剪后的label
        tags_gold: 裁剪后的label
        ignore_labels: 忽略不计的label

    Returns:
        F1, P, R
    """
    assert len(tags_pred) == len(tags_gold)
    true_positive, false_positive, false_negative = 0, 0, 0
    for tag_pred, tag_gold in zip(tags_pred, tags_gold):
        assert len(tag_pred) == len(tag_gold)
        spans_pred = _bio_tag_to_spans(tag_pred, ignore_labels)
        spans_gold = _bio_tag_to_spans(tag_gold, ignore_labels)

        for span in spans_pred:
            if span in spans_gold:
                true_positive += 1
                spans_gold.remove(span)
            else:
                false_positive += 1

        false_negative += len(spans_gold)
    return NERFScore(true_positive, false_positive, false_negative)


if __name__ == '__main__':
    golds = []
    preds = []
    with open('./logs/my_log-2021-01-13-14-27-47/dev.gold.ners', 'r', encoding='utf-8') as gold_reader:
        gold = []
        for line in gold_reader:
            line = line.strip()
            if len(line) == 0:
                golds.append(gold)
                gold = []
            else:
                word, ner = line.split('\t')
                gold.append(ner)

    with open('./logs/my_log-2021-01-13-14-27-47/dev.gold.ners', 'r', encoding='utf-8') as pred_reader:
        pred = []
        for line in pred_reader:
            line = line.strip()
            if len(line) == 0:
                preds.append(pred)
                pred = []
            else:
                word, ner = line.split('\t')
                pred.append(ner)

    res = cal_preformence(preds, golds)
    print(res)
    # res = cal_preformence([['O', 'I-org', 'I-time', 'B-time']], [['O', 'O', 'O', 'O']])
    # print(res)
