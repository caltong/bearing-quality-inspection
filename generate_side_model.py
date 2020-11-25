from torchvision import transforms
from generate_dataset import GenerateDataset
from generate_transform import Generate, ToTensor, ColorJitter, AddBlackBackground, RandomRotation, Flip, Resize
import torch
import time
import copy
import torchvision
import cv2

# 设置opencv 使用单线程 防止dataloader num_workers>0 发生死锁
cv2.setNumThreads(0)
root_dir = './'
train_csv = './train.csv'
val_csv = './val.csv'
SEED = 422
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

epochs = 32
batch_size = 8
lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([Generate(0.5),
                                      ColorJitter(0.5, 1.0, 1.0, 1.0, 1.0),
                                      AddBlackBackground(),
                                      RandomRotation(180),
                                      Flip(0.5),
                                      Resize(512),
                                      ToTensor()])
val_transform = transforms.Compose([Generate(0),
                                    AddBlackBackground(),
                                    Resize(512),
                                    ToTensor()])

train_dataset = GenerateDataset(csv_file='./train.csv', root_dir='./', transform=train_transform)
val_dataset = GenerateDataset(csv_file='./val.csv', root_dir='./', transform=val_transform)
all_data = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
train_dataset, val_dataset = torch.utils.data.random_split(all_data, [len(all_data) - 300, 300])

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

data_loaders = {'train': train_data_loader, 'val': val_data_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}


def train_model(model, criterion, optimizer, scheduler, num_epochs=12):
    print('side model training start')
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample in data_loaders[phase]:
                inputs = sample['image'].to(device)
                labels = sample['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = torchvision.models.resnet101(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = torch.nn.Linear(num_ftrs, 2)

# model_ft = model_ft.to(device)
# 开启多卡
model_ft = torch.nn.DataParallel(model_ft)
model_ft.cuda()

criterion = torch.nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=epochs)

# torch.save(model_ft.state_dict(), 'side_model_use_restnet50_crop_and_crop.pth')
torch.save(model_ft, 'side_model_use_resnet50_and_crop.pth')
