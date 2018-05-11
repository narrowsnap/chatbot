# -*- coding: UTF-8 -*-
import jieba
import re
from gensim.models import word2vec
import numpy as np
# import argparse
#
# parser = argparse.ArgumentParser(description='generate ')
#
# parser.add_argument('-i', '--input_file', dest='source_file', action='store', required=True, help='source file path')
# parser.add_argument('-o', '--segment_output', dest='segment_output',
# action='store', required=True, help='after segment output file path')
# parser.add_argument('-g', '--gensim_output', dest='gensim_output', action='store', required=True, help='after gensim output file path')
# args = parser.parse_args()

PAD = 'PAD'
EOS = 'EOS'
UNK = 'UNK'
GO = 'GO'

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
        inputFile_NoSegment = open(self.segment_input, 'r')
        outputFile_Segment = open(self.segment_output, 'w', encoding='utf-8')

        # 读取语料文本中的每一行文字
        lines = inputFile_NoSegment.readlines()

        # 为每一行文字分词
        for i in range(len(lines)):
            line = lines[i]
            if line:
                line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", line)
                seg_list = jieba.cut(line)

                segments = ''
                for word in seg_list:
                    segments = segments + ' ' + word
                # 为分词后的预料添加结尾
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

    # 将问-答语句输入Q-A中
    def QA(self):
        # 读取分词后的对话文本
        f = open(self.segment_output, 'r', encoding='utf-8')
        subtitles = f.read()

        Q = []
        A = []

        # 将对话文本按段落切分
        subtitles_list = subtitles.split('E')

        # 将"问句"放入Q中，将‘答句’放入A中
        for q_a in subtitles_list:
            # 检验段落中，是否含有‘问-答’句， 如果有，则分别追加到Q和A中
            if re.findall('.*M.*M.*', q_a, flags=re.DOTALL):
                q_a = q_a.strip()
                q_a_pair = q_a.split('M')

                # Q.append(q_a_pair[1].strip())
                # A.append(q_a_pair[2].strip())
                for i in range(len(q_a_pair)):
                    if i == 0:
                        continue
                    else:
                        if i%2 == 1:
                            Q.append(q_a_pair[i].strip())
                        else:
                            A.append(q_a_pair[i].strip())
                if len(Q) > len(A):
                    Q.pop()
                elif len(A) > len(Q):
                    A.pop()

        f.close()

        return Q, A

    # 将Q和A中的词语，转换成词向量，并将问答句长度统一
    def QA_vector(self, Q, A):
        # 导入训练好的词向量
        model = word2vec.Word2Vec.load(self.gensim_output_model)

        # 将Q-A转换为词向量Q_vector、A_vector.format(TEMP_FOLDER)
        Q_vector = []
        for x_sentence in Q:
            x_word = x_sentence.split(' ')

            x_sentvec = [model[w] for w in x_word if w in model.wv.vocab]
            Q_vector.append(x_sentvec)

        A_vector = []
        for y_sentence in A:
            y_word = y_sentence.split(' ')
            y_sentvec = [model[w] for w in y_word if w in model.wv.vocab]
            A_vector.append(y_sentvec)

            # 计算词向量的维数
        word_dim = len(Q_vector[1][0])

        # 设置结束词
        sentend = np.ones((word_dim,), dtype=np.float32)

        # 将问-答句的长度统一
        for sentvec in Q_vector:
            if len(sentvec) > 14:
                # 将第14个词之后的全部内容删除，并将第15个词换为sentend
                sentvec[14:] = []
                sentvec.append(sentend)
            else:
                # 将不足15个词的句子，用sentend补足
                for i in range(15 - len(sentvec)):
                    sentvec.append(sentend)

        for sentvec in A_vector:
            if len(sentvec) > 15:
                sentvec[14:] = []
                sentvec.append(sentend)
            else:
                for i in range(15 - len(sentvec)):
                    sentvec.append(sentend)

        return Q_vector, A_vector

    # 将QA vector 拆分成小文件
    def separate_qa_vector(self, q_v, a_v):
        count = 0
        for i in range(len(q_v)):
            if i % 2 == 0:
                np.save('data/train/q_'+str(count)+'.npy', q_v[count*2:i])
                np.save('data/train/a_'+str(count)+'.npy', a_v[count*2:i])
                count += 1


def test():
    word = Word(
        'data/dgk_shooter.conv',
        'data/dgk_segment.conv',
        'data/dgk_segment.conv',
        'model/dgk_gensim_model'
    )
    # word.word_segment()
    # word.word_model()
    # Q, A = word.QA()
    # word.deal_with_qa(Q, A)
    # q_v, a_v = word.QA_vector(Q, A)
    q_v = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [1, 2, 3], [2, 3, 4], [3, 4, 5], [3, 4, 5]]
    a_v = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [1, 2, 3], [2, 3, 4], [3, 4, 5], [3, 4, 5]]
    q_v = np.array(q_v)
    a_v = np.array(a_v)
    word.piece_qa_vector(q_v, a_v)

if __name__ == '__main__':
    test()