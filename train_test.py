import torch
import data_preprocess
import torch.nn as nn
import gc
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import nnmdefine
import numpy as np

# data prarameters
concat_nframes = 1              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.8               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 0                        # random seed
batch_size = 512                # batch size
num_epoch = 5                   # the number of training epoch
learning_rate = 0.0001          # learning rate
model_path = './model.ckpt'     # the path where the checkpoint will be saved

# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 1               # the number of hidden layers
hidden_dim = 256                # the hidden dim

train_x,train_y=data_preprocess.preprocess_data(split='train',
                                                feat_dir='./libriphone/feat',
                                                phone_path='./libriphone',
                                                concat_nframes=concat_nframes,
                                                train_ratio=train_ratio)
val_x,val_y=data_preprocess.preprocess_data(split='val',
                                            feat_dir='./libriphone/feat',
                                            phone_path='./libriphone',
                                            concat_nframes=concat_nframes,
                                            train_ratio=train_ratio
                                            )
train_set=data_preprocess.LibriDataset(train_x,train_y)
val_set=data_preprocess.LibriDataset(val_x,val_y)

del train_x,train_y,val_y,val_x
gc.collect()

train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

same_seed(seed)

model = nnmdefine.Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

best_acc=0.0
writer=SummaryWriter('logs')
step=0
for epoch in range(num_epoch):
    train_acc=0.0
    train_loss=0.0
    val_acc=0.0
    val_loss=0.0
    model.train()
    for i,batch in enumerate(tqdm(train_loader)):
        features,labels = batch
        features,labels = features.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs=model(features)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        step+=1
        _,train_pred = torch.max(outputs,1)
        train_acc +=(train_pred.detach()==labels.detach()).sum().item()
        train_loss+=loss.item()

    if len(val_set)>0:
        model.eval()
        with torch.no_grad():
            for i,batch in enumerate(tqdm(val_loader)):
                features,labels=batch
                features,labels=features.to(device),labels.to(device)
                outputs=model(features)
                loss=criterion(outputs,labels)
                _,val_pred=torch.max(outputs,1)
                val_acc+=(val_pred.cpu()==labels.cpu()).sum().item()
                val_loss+=loss.item()
            print ('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format (
                epoch + 1, num_epoch, train_acc / len (train_set), train_loss / len (train_loader),
                val_acc / len (val_set), val_loss / len (val_loader)
            ))

            if val_acc>best_acc:
                best_acc=val_acc
                torch.save(model.state_dict(),model_path)
                print ('saving model with acc {:.3f}'.format (best_acc / len (val_set)))
    else:
        print ('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format (
            epoch + 1, num_epoch, train_acc / len (train_set), train_loss / len (train_loader)
        ))
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

del train_loader, val_loader
gc.collect()


test_X = data_preprocess.preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
test_set =data_preprocess.LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
model = nnmdefine.Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
test_acc=0.0
test_lenths=0
model.load_state_dict(torch.load('./model.ckpt'))
pred = np.array([],dtype=np.int32)
model.eval()
with torch.no_grad():
    for i,batch in enumerate(tqdm(test_loader)):
        features=batch
        features=features.to(device)
        outputs=model(features)
        _,test_pred =torch.max(outputs,1)
        pred = np.concatenate((pred,test_pred.cpu().numpy()),axis=0)


with open('prediction.csv','w') as f:
    f.write('id,Class\n')
    for i,y in enumerate(pred):
        f.write(f'{i},{y}\n')