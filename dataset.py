import torch.utils.data as data
import torch
import sys
import numpy as np

class VecDataSet(data.Dataset):
    def __init__(self, q, a):   # question, answer
        self.q = q
        self.a = a

    def __getitem__(self, index):
        q, a = self.q[index], self.a[index]
        return q, a

    def __len__(self):
        return len(self.q)

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
    q_v, a_v = w.XY_vector(q, a)
    test(q_v, a_v)