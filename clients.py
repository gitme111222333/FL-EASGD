import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet


class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None


    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        # if self.local_parameters==None:
        #     Net.load_state_dict(global_parameters, strict=True)
        #
        # else:
        #     for var in global_parameters:
        #         self.local_parameters[var] = self.local_parameters[var] - 0.8 * ( self.local_parameters[var] - global_parameters[var])
        #     Net.load_state_dict(self.local_parameters, strict=True)
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        # 数据集名称，是否独立同分布，参与方个数，GPU or CPU
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        # clients_set格式为'client{i}' : Client(i)
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    # 初始化CLientGroup内容
    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        # 测试集数据和标签（标签由向量转换为整型，如[0,0,1]->2）
        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        # 训练集数据和标签
        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label
        self.train_data_loader = DataLoader(
            TensorDataset(torch.tensor(train_data), torch.argmax(torch.tensor(train_label), dim=1)), batch_size=100,
            shuffle=False)

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        # 初始化num_of_clients，为每个参与方分配数据
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)
            # 生成client，训练测试数据由np转为tensor
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            self.clients_set['client{}'.format(i)] = someone

# 测试ClientGroup
# if __name__=="__main__":
#     MyClients = ClientsGroup('mnist', True, 100, 1)
#     print(MyClients.clients_set['client10'].train_ds[0:100])
#     print(MyClients.clients_set['client11'].train_ds[400:500])


