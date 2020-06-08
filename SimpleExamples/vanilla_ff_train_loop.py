from torch.nn import init
from torch import nn
from torch import optim
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
from torch import tensor
import torch.nn.functional as F

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'

def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)

def test_near(a,b): test(a,b,near)

def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s): return (x-m)/s

def test_near_zero(a,tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"

def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()

def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

def get_model():
    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
    return model, optim.SGD(model.parameters(), lr=lr)

class Dataset():
    def __init__(self, x, y): self.x,self.y = x,y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i],self.y[i]

class Sampler():
    def __init__(self, ds, bs, shuffle=False):
        self.n,self.bs,self.shuffle = len(ds),bs,shuffle
        
    def __iter__(self):
        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs): yield self.idxs[i:i+self.bs]

def collate(b):
    xs,ys = zip(*b)
    return torch.stack(xs),torch.stack(ys)

class DataLoader():
    def __init__(self, ds, sampler, collate_fn=collate):
        self.ds,self.sampler,self.collate_fn = ds,sampler,collate_fn
    def __len__(self): return len(self.ds)   
    def __iter__(self):
        for s in self.sampler: yield self.collate_fn([self.ds[i] for i in s])

# def get_dls(train_ds, valid_ds, bs, **kwargs):
#     train_samp = Sampler(train_ds, bs, shuffle=True)
#     valid_samp = Sampler(valid_ds, bs, shuffle=False)
#     return (DataLoader(train_ds, sampler=train_samp, collate_fn=collate),
#             DataLoader(valid_ds, sampler=valid_samp, collate_fn=collate))

def get_dls(train_ds, valid_ds, bs, **kwargs):
    from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # Handle batchnorm / dropout
        model.train()
        for xb,yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in valid_dl:
                pred = model(xb)
                tot_loss += loss_func(pred, yb)
                tot_acc  += accuracy (pred,yb)
        nv = len(valid_dl)
        print(epoch, tot_loss/nv, tot_acc/nv)
    return tot_loss/nv, tot_acc/nv

x_train,y_train,x_valid,y_valid = get_data()
n,m = x_train.shape
c = y_train.max()+1
nh = 50
lr = 0.5   # learning rate
loss_func = F.cross_entropy
train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)
train_dl,valid_dl = get_dls(train_ds, valid_ds, 64)
model,opt = get_model()
loss,acc = fit(5, model, loss_func, opt, train_dl, valid_dl)