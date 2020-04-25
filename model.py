import torch
import torchvision


def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = torch.nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    return model_ft
