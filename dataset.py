import torch.utils.data as data
import torch
import sys

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
    if(q):
        dataset = VecDataSet(q, a)
        kwargs = {'num_workers': 1, 'pin_memory': True} if sys.args.cuda else {}
        train_loader = data.DataLoader(dataset, batch_size=64, shuffle=True, **kwargs)
        print(train_loader)
    else:
        print('not implement')

if __name__ == '__main__':
    test()