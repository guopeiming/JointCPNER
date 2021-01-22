

if __name__ == '__main__':
    with open('./data/pretrain/zh.par.corpus', 'w', encoding='utf-8') as writer:
        for i in range(5):
            with open('./data/pretrain/'+str(i+1)+'.corpus.par.out', 'r', encoding='utf-8') as reader:
                for line in reader:
                    writer.write(line)
