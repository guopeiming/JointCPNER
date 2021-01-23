from collections import Counter

if __name__ == '__main__':
    counter = Counter()
    with open('./data/pretrain_char/token.vocab', 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            word, num = line.split('\t')
            counter[word] = int(num)

    print(counter)

