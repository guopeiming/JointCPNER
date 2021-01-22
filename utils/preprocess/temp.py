

if __name__ == '__main__':
    with open('./SAPar/data_processing/dev.ctb60.cor', 'r', encoding='utf-8') as reader,\
         open('./SAPar/data_processing/dev.corpus', 'w', encoding='utf-8') as writer:
        for line in reader:
            line = line.strip()
            assert line[:6] == '(ROOT ' and line[-1] == ')'
            writer.write(line[6:-1]+'\n')
