import torch
from data import UpperAndLowerFacesData, Rescale, ToTensor
import torchvision

upper_and_lower_dataset = UpperAndLowerFacesData(torchvision.transforms.Compose([Rescale(224), ToTensor()]))
upper_and_lower_data_loader = torch.utils.data.DataLoader(upper_and_lower_dataset, batch_size=32, shuffle=True)

# for i_batch, sample_batched in enumerate(upper_and_lower_data_loader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['label'].size())

model = torchvision.models.vgg16_bn(pretrained=False)
print(model.features.children())
# print(model)

# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in enumerate(upper_and_lower_data_loader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data['image'], data['label']
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         print('[%d, %5d] loss: %.3f' %
#               (epoch + 1, i + 1, running_loss / 2000))
#         running_loss = 0.0
#
# print('Finished Training')
