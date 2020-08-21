import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from vgg import vgg
import numpy as np

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model = vgg()
if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print(model)
#计算scale个数
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]
#将所有scale值拷贝到bn中
bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size
#排序，根据要裁剪的比例计算筛选阈值
y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]
#根据阈值将小于阈值的各通道参数全部处理设置成0
pruned = 0
cfg = [] ##记录保留通道数
cfg_mask = []#记录保留通道的位置
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d): ##原则上batchnorm在conv之后
        weight_copy = m.weight.data.clone()
        thre=thre.cuda()
        mask = weight_copy.abs().gt(thre).float().cuda()
        #mask = torch.gt(weight_copy.abs(),thre).float().cuda()#torch.gt逐个元素比较,mask记录需要保存的通道，保存通道为1，不保存的为0
        ## 待改进1 ：裁剪后通道数是8的整数倍、不能整层裁剪掉###########################
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)#将需要删除的通道进行掩模处理为0,模型就已经被修改了
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask))) #内部记录需要保留通道的数目，只有用 batchnorm时才有值，加入有卷积之后没有batchnorm则可能有问题，但VGG不存在这种情况
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total ##计算裁剪掉的比例

print('Pre-processing Successful!')

#测试将低于阈值的通道抹除之后的测试精度
# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test():
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {} #pin_memory将数据保存在pin转gpu速度更快些，num_workers：控制使用多进程，0代表不使用；
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

test()

##如果要做通用的话，外部只需要把网络结构、正则化训练的模型 传进来，代码自动遍历裁剪

# Make real prune
print(cfg)
newmodel = vgg(cfg=cfg)#####这样使用要求每一个卷积之后都有batchnorm,否则网络对不起来；这里相当于新建了一个网络结构
newmodel.cuda()

#主要记录如何将权值拷贝过来，针对batchnorm、卷积、全连接层
layer_id_in_cfg = 0
start_mask = torch.ones(3) # 为何是3？是第一层卷积的输入通道数
end_mask = cfg_mask[layer_id_in_cfg]#cfg_mask记录了需要保留的通道序号
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))#np.argwhere--返回非0的数组元组的索引，np.squeeze--即把shape中为1的维度去掉
        m1.weight.data = m0.weight.data[idx1].clone()
        m1.bias.data = m0.bias.data[idx1].clone()
        m1.running_mean = m0.running_mean[idx1].clone()
        m1.running_var = m0.running_var[idx1].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d): # 默认卷积层都能紧挨着BN层，且在BN层之前
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy()))) ## 默认每隔卷积都挨着BN层
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
        w = m0.weight.data[:, idx0, :, :].clone() #idx0 是输入保存通道的list，注意权值的第一个值对应输出序号，第二个对应输入序号；
        w = w[idx1, :, :, :].clone() #idx1 是输出保存通道的list
        m1.weight.data = w.clone()
        # m1.bias.data = m0.bias.data[idx1].clone()
    elif isinstance(m0, nn.Linear): #默认全连接层全部在BN层之后
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        m1.weight.data = m0.weight.data[:, idx0].clone()


torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)

print(newmodel)
model = newmodel
test()