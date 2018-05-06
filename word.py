# -*- coding: UTF-8 -*-
import jieba
import re
from gensim.models import word2vec
import numpy as np

class Word:
    def __init__(self,
                 segment_input,
                 segment_output,
                 gensim_input_file,
                 gensim_output_model
                 ):
        self.segment_input = segment_input
        self.segment_output = segment_output
        self.gensim_input_file = gensim_input_file
        self.gensim_output_model = gensim_output_model

    # 为语料做分词处理
    def word_segment(self):

        # 打开语料文本
        inputFile_NoSegment = open(self.segment_input, 'rb')
        outputFile_Segment = open(self.segment_output, 'w', encoding='utf-8')

        # 读取语料文本中的每一行文字
        lines = inputFile_NoSegment.readlines()

        # 为每一行文字分词
        for i in range(len(lines)):
            line = lines[i]
            if line:
                line = line.strip()
                seg_list = jieba.cut(line)

                segments = ''
                for word in seg_list:
                    segments = segments + ' ' + word
                # 为分词后的预料添加结尾
                segments = segments + ' eos'
                segments += '\n'
                segments = segments.lstrip()

                # 将分词后的语句，写进文件中
                outputFile_Segment.write(segments)

        inputFile_NoSegment.close()
        outputFile_Segment.close()


    # 训练gensim模型
    def word_model(self):
        sentences = word2vec.Text8Corpus(self.gensim_input_file)
        model = word2vec.Word2Vec(sentences, min_count=5, size=100)
        model.save(self.gensim_output_model)

    # 将问-答语句输入X-Y中
    def XY(self):
        # 读取分词后的对话文本
        f = open(self.segment_output, 'r', encoding='utf-8')
        subtitles = f.read()

        X = []
        Y = []

        # 将对话文本按段落切分
        subtitles_list = subtitles.split('E')

        # 将"问句"放入X中，将‘答句’放入Y中
        for q_a in subtitles_list:
            # 检验段落中，是否含有‘问-答’句， 如果有，则分别追加到X和Y中
            if re.findall('.*M.*M.*', q_a, flags=re.DOTALL):
                q_a = q_a.strip()
                q_a_pair = q_a.split('M')

                # X.append(q_a_pair[1].strip())
                # Y.append(q_a_pair[2].strip())
                for i in range(len(q_a_pair)):
                    if i == 0:
                        continue
                    else:
                        if i%2 == 1:
                            X.append(q_a_pair[i].strip())
                        else:
                            Y.append(q_a_pair[i].strip())
                if len(X) > len(Y):
                    X.pop()
                elif len(Y) > len(X):
                    Y.pop()

        f.close()

        return X, Y

    # 将X和Y中的词语，转换成词向量，并将问答句长度统一
    def XY_vector(X, Y):

        # 导入训练好的词向量
        model = word2vec.Word2Vec.load(self.gensim_output_model)

        # 将X-Y转换为词向量X_vector、Y_vector.format(TEMP_FOLDER)
        X_vector = []
        for x_sentence in X:
            x_word = x_sentence.split(' ')

            x_sentvec = [model[w] for w in x_word if w in model.wv.vocab]
            X_vector.append(x_sentvec)

        Y_vector = []
        for y_sentence in Y:
            y_word = y_sentence.split(' ')
            y_sentvec = [model[w] for w in y_word if w in model.wv.vocab]
            Y_vector.append(y_sentvec)

        # 计算词向量的维数
        word_dim = len(X_vector[1][0])

        # 设置结束词
        sentend = np.ones((word_dim,), dtype=np.float32)

        # 给问-答句添加结尾
        for sentvec in X_vector:
            sentvec.append(sentend)

        for sentvec in Y_vector:
            sentvec.append(sentend)

        return X_vector, Y_vector

def test():
    word = Word(
        'data/dgk_shooter.conv',
        'data/dgk_segment.conv',
        'data/dgk_segment.conv',
        'model/dgk_gensim_model'
    )
    word.word_segment()
    word.word_model()


if __name__ == '__main__':
    test()