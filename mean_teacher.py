import torch
import math
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import KFold


from torch.utils.tensorboard import SummaryWriter
from braindecode.util import set_random_seeds
from tqdm import tqdm
from library.train_loop import TrainLoop
from library.optmization import Optmization
from library.optmization import ema_model, WeightEMA
from library.eegconformer import EEGConformer
from data.dataloader import load_eeg_data

# 超参数设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = 3
num_epochs = 20
batch_size = 12
learning_rate = 0.0001
strong_aug = 0.8
weak_aug = 0.2
lambda_u = 0.5
T = 0.8
alpha = 0.75
init_weight = 0
end_weight = 30
w_da = 1.0
ema_decay = 0.99



# 文件列表
file_list = [
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/1_20131027.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/1_20131030.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/1_20131107.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/2_20140404.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/2_20140413.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/2_20140419.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/3_20140603.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/3_20140611.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/3_20140629.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/4_20140621.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/4_20140702.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/4_20140705.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/5_20140411.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/5_20140418.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/5_20140506.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/6_20130712.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/6_20131016.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/6_20131113.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/7_20131027.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/7_20131030.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/7_20131106.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/8_20140511.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/8_20140514.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/8_20140521.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/9_20140620.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/9_20140627.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/9_20140704.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/10_20131130.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/10_20131204.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/10_20131211.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/11_20140618.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/11_20140625.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/11_20140630.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/12_20131127.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/12_20131201.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/12_20131207.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/13_20140527.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/13_20140603.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/13_20140610.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/14_20140601.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/14_20140615.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/14_20140627.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/15_20130709.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/15_20131016.mat',
    '/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/15_20131105.mat'
]

raw_eeg = load_eeg_data(file_list)
label = io.loadmat('/home/user/data/EEG_emotion/SEED/SEED/Preprocessed_EEG/label.mat')
#定义窗口长度
len_window = 200 * 4

# 初始化存储列表
raw_X = []
raw_y = []

# 前缀列表
prefix = ['djc', 'jl', 'jj', 'lqj', 'ly', 'mhw', 'phl', 'sxy', 'wk', 'ww', 'wsf', 'wyw', 'xyl', 'ys', 'zjy']
sub = 0

def process_element(element, prefix, sub, len_window, label, raw_X, raw_y):
    for i in range(1, 16):
        data = element[prefix[sub] + "_eeg" + str(i)]
        n_windows = data.shape[1] // len_window
        reshaped_X = np.reshape(data[:, :n_windows * len_window], (62, len_window, n_windows))
        raw_X.append(reshaped_X)
        raw_y.append(np.array([label['label'][0][i-1] for _ in range(n_windows)]))

while raw_eeg and sub < 15:  # only using the first 12 subjects to avoid memory exceed
    for _ in range(3):  # Process each experiment 3 times
        element = raw_eeg.pop(0)
        process_element(element, prefix, sub, len_window, label, raw_X, raw_y)
        del element
    sub += 1

concat_X = np.concatenate(raw_X, axis=2)
X = concat_X.transpose((2, 0, 1))
concat_y = np.concatenate(raw_y)

# 标签编码
concat_y = np.concatenate(raw_y)
le = LabelEncoder()
y = le.fit_transform(concat_y)
y = pd.get_dummies(y)

# 5-fold 交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_test_accuracies = []
train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []

seed = 20240216
set_random_seeds(seed=seed, cuda=device)

# 定义EEG-conformer model
class Model(nn.Module):
    def __init__(self, EEGConformer):
        super(Model, self).__init__()
        self.model = EEGConformer(
                    n_outputs=3,
                    n_chans=62,
                    n_times=800, # input_winodw_samples
                    input_window_seconds=4,
                    sfreq=200,
                )

    def augmentation(self, input, std):

        input_shape =input.size()
        noise = torch.normal(mean=0.5, std=std, size =input_shape)
        noise = noise.to(device)

        return input + noise

    def forward(self, input, compute_model=True):

        if compute_model==False:
            input_s  = self.augmentation(input, strong_aug)
            input_w  = self.augmentation(input, weak_aug)

            output = (input_s, input_w)

        else:
            output = self.model(input)
        return output
    
# 定义Teacher模型
class EMA_Model(nn.Module):
    def __init__(self, EEGConformer):
        super(EMA_Model, self).__init__()
        self.model = EEGConformer(
                    n_outputs=3,
                    n_chans=62,
                    n_times=800, # input_winodw_samples
                    input_window_seconds=4,
                    sfreq=200,
                )

    def forward(self, input):
        return self.model(input)

input_size = 62 
hidden_size = 256
output_size = 3

# 定义EMA更新函数
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def train(model, train_labeled_loader, train_unlabeled_loader, test_loader, save_path='./model_transformer/'):
    writer = SummaryWriter('../log')
    total_step = len(train_labeled_loader)
    best_accuracy = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    unsupervised_weight = 1

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm(total=total_step, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', miniters=1) as pbar:
            for (inputs_x, targets_x), (inputs_u, targets_u_true) in zip(train_labeled_loader, train_unlabeled_loader):
                inputs_x = inputs_x.to(device)
                targets_x = targets_x.to(device)
                inputs_u = inputs_u.to(device)

                optimizer.zero_grad()
                outputs = model(inputs_x)
                loss_x = criterion(outputs, targets_x)

                # 获取Teacher模型的预测
                with torch.no_grad():
                    outputs_u_teacher = ema_model(inputs_u)
                
                outputs_u_student = model(inputs_u)
                loss_u = F.mse_loss(outputs_u_student, outputs_u_teacher)

                unsupervised_weight =  lambda_u * unsupervised_weight
                loss = loss_x + unsupervised_weight * loss_u
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                update_ema_variables(model, ema_model, ema_decay, global_step)
                global_step += 1

                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
            pbar.set_postfix({'epoch_loss': epoch_loss / total_step})

        scheduler.step()
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                output = model(features)
                _, predicted = torch.max(output.data, 1)
                _, label = torch.max(labels, 1)
                total += labels.size(0)
                correct += (predicted == label).sum().item()

            test_accuracy = correct / total
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                if save_path is not None:
                    torch.save(model.state_dict(), save_path + 'best_model.pth')
                    print("best_model found, best acc: ", best_accuracy)

        print('Test Accuracy is {}%'.format(100 * correct / total))

    if save_path is not None: 
        model.load_state_dict(torch.load(save_path + 'best_model.pth'))

    writer.close()
    return best_accuracy


for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f'Fold {fold + 1}')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    x_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train.values).float().to(device)
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(x_train_tensor, y_train_tensor, test_size=0.8, random_state=42)
    
    train_labeled_loader = DataLoader(TensorDataset(X_labeled, y_labeled), batch_size=batch_size, shuffle=True)
    train_unlabeled_loader = DataLoader(TensorDataset(X_unlabeled, y_unlabeled), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test.values).float().to(device)), batch_size, shuffle=False)

    model = Model(EEGConformer).to(device)
    ema_model = EMA_Model(EEGConformer).to(device)
    for param in ema_model.parameters():
        param.requires_grad = False
    best_accuracy = train(model, train_labeled_loader, train_unlabeled_loader, test_loader, save_path='./model_transformer/')
    all_test_accuracies.append(best_accuracy)

mean_test_accuracy = np.mean(all_test_accuracies)
std_error = np.std(all_test_accuracies) / math.sqrt(kf.get_n_splits())
print(f'Mean Test Accuracy across all folds: {100 * mean_test_accuracy:.2f}% ± {100 * std_error:.2f}%')



