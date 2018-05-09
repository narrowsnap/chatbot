import torch.utils.data as data
import torch
import sys
import numpy as np
from gensim.models import word2vec

class VecDataSet(data.Dataset):
    def __init__(self, question, answer, gensim_model, transform):
        self.question = question
        self.answer = answer
        self.model = word2vec.Word2Vec.load(gensim_model)
        self.transform = transform

    def __getitem__(self, index):
        q, a = self.question[index], self.answer[index]
        q_vector = []
        for q_sentence in q:
            q_word = q_sentence.split(' ')

            q_sentvec = [self.model[w] for w in q_word if w in self.model.wv.vocab]
            q_vector.append(q_sentvec)

        a_vector = []
        for a_sentence in a:
            a_word = a_sentence.split(' ')
            a_sentvec = [self.model[w] for w in a_word if w in self.model.wv.vocab]
            a_vector.append(a_sentvec)
        return self.transform(np.array(q_vector)), self.transform(np.array(a_vector))

    def __len__(self):
        return len(self.question)

def test(q, a):
    dataset = VecDataSet(q, a)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    # kwargs = {'num_workers': 1, 'pin_memory': True} if sys.args.cuda else {}
    train_loader = data.DataLoader(dataset, batch_size=64, shuffle=True, **kwargs)
    print(train_loader.shape)

if __name__ == '__main__':
    import word
    w = word.Word(
        'data/dgk_shooter.conv',
        'data/dgk_segment.conv',
        'data/dgk_segment.conv',
        'model/dgk_gensim_model'
    )
    q, a = w.XY()
    test(q, a)