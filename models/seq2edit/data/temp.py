def mix_vocab(vocab_path1, vocab_path2):
    with open(vocab_path2, "r", encoding="utf-8") as f2:
        data2 = [line for line in f2.read().split("\n") if line]
    with open(vocab_path1, "r", encoding="utf-8") as f1, open(vocab_path2, "a", encoding="utf-8") as f2:
        data1 = [line for line in f1.read().split("\n") if line]
        for line in data1:
            if line not in data2:
                f2.write(line+"\n")


def analyse(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = [line for line in f.read().split("\n") if line]
        append_count, replace_count = 0, 0
        for line in data:
            if "$APPEND" in line:
                append_count += 1
            elif "$REPLACE" in line:
                replace_count += 1
            else:
                print(line)
        print(append_count, replace_count)



vocab_path1 = "output_vocabulary_chinese_char_hsk+lang8_5/labels.txt"
vocab_path2 = "output_vocabulary_chinese_native/labels.txt"
mix_vocab(vocab_path1, vocab_path2)

analyse(vocab_path1)
