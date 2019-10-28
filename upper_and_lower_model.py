import torch
from data import UpperAndLowerFacesData, Rescale, RandomCrop, ToTensor
import torchvision

epochs = 32
batch_size = 16
lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

upper_and_lower_dataset = UpperAndLowerFacesData(torchvision.transforms.Compose([Rescale(256),
                                                                                 # RandomCrop(224),
                                                                                 ToTensor()]))
upper_and_lower_data_loader = torch.utils.data.DataLoader(upper_and_lower_dataset, batch_size=batch_size, shuffle=True)

# for i_batch, sample_batched in enumerate(upper_and_lower_data_loader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['label'].size())

model = torchvision.models.vgg16_bn(pretrained=True)
model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=1, bias=True)
model = torch.nn.Sequential(model, torch.nn.Sigmoid())
# print(model.features.children())
# print(model)
model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):  # loop over the dataset multiple times
    loss_one_epoch = 0
    for i, data in enumerate(upper_and_lower_data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'].to(device), data['label'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_one_epoch += (loss.item() * len(labels))
        loss.backward()
        optimizer.step()

        # print statistics
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in upper_and_lower_data_loader:
    #         images, labels = data['image'].to(device), data['label'].to(device)
    #         outputs = model(images)
    #         # print(outputs)
    #         for i in range(outputs.size()[0]):
    #             if outputs[i] > 0.5 and labels[i] == 1:
    #                 correct += 1
    #                 total += 1
    #             elif outputs[i] <= 0.5 and labels[i] == 0:
    #                 correct += 1
    #                 total += 1
    #             else:
    #                 total += 1
    # print('Accuracy of the network on the 132 test images: %d %%' % (
    #         100 * correct / total))
    print('Loss of the network on the 132 test images: %f' % (
            loss_one_epoch / len(upper_and_lower_dataset)))

print('Finished Training')
