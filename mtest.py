import torch
import torch.nn as nn
import tenseal as ts
import torch.optim as optim
import torchvision
import numpy as mp
import matplotlib.pyplot as plt
import torch.nn.functional as F


# define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # Initialize model
    model = TheModelClass()

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40
    # print model's state_dict
    print('Model.state_dict:')
    sum_parameters = None
    parameters_shape = None
    # print(model.state_dict())
    for param_tensor in model.state_dict():

        if sum_parameters is None:
            sum_parameters = {}
            parameters_shape = {}

        # print(model.state_dict().keys()) #odict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias'])
        # print(param_tensor) # keys: 各层的weight和bias
        # 打印 key value字典
        # print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        plain1 = model.state_dict()[param_tensor]
        parameters_shape[param_tensor] =plain1.shape
        plain1 = plain1.flatten(0).cpu().numpy().tolist()
        # print(" Encrypted Data = {}.".format(plain1))
        encrypted_tensor1 = ts.ckks_tensor(context, plain1)
        print(" Shape = {}".format(encrypted_tensor1.shape))
        # print(param_tensor, '\t', " Encrypted Data = {}.".format(encrypted_tensor1))
        sum_parameters[param_tensor] = encrypted_tensor1
    # print(sum_parameters)
    def decrypt(enc):
        return enc.decrypt().tolist()
    global_parameters = {}
    for var in sum_parameters:
        # sum_parameters[var] = torch.reshape(torch.Tensor(sum_parameters[var]), parameters_shape[var])
        global_parameters[var] = (sum_parameters[var] / torch.tensor(2.0))
        print(global_parameters[var].shape)
    print((global_parameters))
    for var in global_parameters:
        global_parameters[var] = decrypt(global_parameters[var])
        global_parameters[var] = torch.Tensor(global_parameters[var])
        print(global_parameters[var].shape)
        global_parameters[var] = torch.reshape(torch.Tensor(global_parameters[var]), parameters_shape[var])
        print("===================")
        print(global_parameters[var].shape)
    #
    print(global_parameters)
    # model.load_state_dict(global_parameters, strict=True)
    # print(model.state_dict())


    # encrypted_tensor1 / torch.tensor(2.0)
    # # print optimizer's state_dict
    # print('Optimizer,s state_dict:')
    # for var_name in optimizer.state_dict():
    #     print(var_name, '\t', optimizer.state_dict()[var_name])


if __name__ == '__main__':
    main()

