import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from utils.options import args_parser
from models.Nets import LeNet

import data_loading as dataset

def get_train_valid_loader(data_dir,
                           batch_size,
                           num_workers=0,
                           ):
    # Create Transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])

    # Create Datasets
    dataset_train = dataset.TrafficSignData(
        root_dir=data_dir, train=True,  transform=transform)
    dataset_test = dataset.TrafficSignData(
        root_dir=data_dir, train=False,  transform=transform)

    # Load Datasets
    return dataset_train, dataset_test

def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Load training and testing datasets.
    torch.manual_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'traffic':
        dataset_train, dataset_test = get_train_valid_loader('/home/namphuong/Code/ImageProcessing/advanced-image-processing-cau/traffic-sign-recognition/data',
                                                                         batch_size=32, num_workers=0)
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'LeNet' and args.dataset == 'traffic':
        net_glob = LeNet(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=10, shuffle=True, num_workers=0)

    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

    # plot loss
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('./results/centralize_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    # testing
    if args.dataset == 'traffic':
        dataset_train, dataset_test = get_train_valid_loader(
            '/home/namphuong/Code/ImageProcessing/advanced-image-processing-cau/traffic-sign-recognition/data',
            batch_size=32, num_workers=0)
        test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=10, shuffle=False, num_workers=0)
    else:
        exit('Error: unrecognized dataset')

    print('test on', len(dataset_test), 'samples')
    test_acc, test_loss = test(net_glob, test_loader)
