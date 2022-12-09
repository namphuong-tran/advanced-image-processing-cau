import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import transforms
import torch

from utils.sampling import traffic_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import LeNet
from models.Federated import FedAvg
from models.evaluate import evaluate
import data_loading as dataset
import time




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

if __name__ == '__main__':
    start_time = time.time()
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load dataset and split users
    if args.dataset == 'traffic':
        dataset_train, dataset_test = get_train_valid_loader(
            '/home/namphuong/Code/ImageProcessing/advanced-image-processing-cau/traffic-sign-recognition/data',
            batch_size=32, num_workers=0)
        if args.iid:
            dict_users = traffic_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in Traffic')
        
    #img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'LeNet' and args.dataset == 'traffic':
        net_glob = LeNet(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./results/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = evaluate(net_glob, dataset_train, args)
    acc_test, loss_test = evaluate(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    print("--- %s seconds ---" % (time.time() - start_time))