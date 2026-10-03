import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import os
import struct
import numpy as np

def jis_to_char(code): #converts jis code to character 
    
    return bytes([(code >> 8) | 0x80, (code & 0xFF) | 0x80]).decode('euc_jp')

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding =1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)

        self.fc1 = nn.Linear(7*8*128,512)
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.max_pool2d(F.relu(self.conv3(x)),2)
        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split 

    all_labels = []
    all_matrices = []

    torch.manual_seed(67)

    for path in ['ETL8B2C1', 'ETL8B2C2', 'ETL8B2C3']:
        with open(path, 'rb') as file:
            file.read(512)
            while True:
                data = file.read(512)
                if not data:
                    break
                data_unpack = struct.unpack('> H H 4x 504s', data)
                byte_array = np.frombuffer(data_unpack[2], dtype=np.uint8)
                all_matrices.append(np.unpackbits(byte_array).reshape(63, 64))
                all_labels.append(data_unpack[1])


    unique_labels = sorted(set(all_labels)) #remove duplicates and sort
    label_index = {label: i for i, label in enumerate(unique_labels)} # dictionary of items label: i.

    y_data = np.array([label_index[l] for l in all_labels]) #already in the correct order
    X_data = np.array(all_matrices)

    y_tensor = torch.from_numpy(y_data).long()
    X_tensor = torch.from_numpy(X_data)  # keep as uint8

    X_tensor = X_tensor.unsqueeze(1)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size = 0.2, random_state=67)

    del X_tensor, X_data, all_matrices  # X_tensro and X_data share same memory

    train_ds = TensorDataset(X_train,y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=256)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(len(unique_labels)).to(device)

    # reuse the saved model if exists
    if os.path.exists('model.pt'):
        model.load_state_dict(torch.load('model.pt', map_location=device)['model'])
        print('loaded model.pt, skipping training')
    else:
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr = 1e-3)

        for epoch in range(10):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.float().to(device), yb.to(device)
                opt.zero_grad()                 # clear last batch's gradients
                loss = loss_fn(model(xb), yb)   # forward + loss
                loss.backward()                 # gradients
                opt.step()                      # change 

            model.eval()
            correct = 0
            with torch.no_grad():
                for xb, yb in test_dl:
                    preds = model(xb.float().to(device)).argmax(1).cpu()
                    correct += (preds == yb).sum().item()
            print(f'epoch {epoch+1}: test acc {correct / len(y_test):.4f}')

        # save weights 
        torch.save({'model': model.state_dict(), 'unique_labels': unique_labels}, 'model.pt')
       
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_dl:
            all_preds.append(model(xb.float().to(device)).argmax(1).cpu())
    all_preds = torch.cat(all_preds)

    wrong = (all_preds != y_test).nonzero().flatten()
    print(f'{len(wrong)} of {len(y_test)} test images misclassified')

