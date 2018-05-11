import torch.utils.data as data
import numpy as np

class VecDataSet(data.Dataset):
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer

    def __getitem__(self, index):
        q, a = self.question[index], self.answer[index]
        return q, a

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