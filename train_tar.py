import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from scipy.spatial.distance import cdist
def Entropy(input_ ,dim):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=dim)
    return entropy
def newre(softmax_f, softmax, min,pred):
    bs = softmax_f.size(0)
    for i in range(bs):
        #print(softmax_f[i])

        if Entropy(softmax_f[i],0) <= args.kl and min[i] == 0:continue
        elif Entropy(softmax[i],0) <= args.kl :
            softmax_f[i] = softmax[i]
            min[i]=args.batch_size+2
        else :
            softmax_f[i] = pred[i]
            min[i]=args.batch_size+1
    return softmax_f,min

def fill_tensor_based_on_percentile(tensor, r):
    # 计算要取的位置
    k = int(tensor.size(1) * r)
    if (k < 1): k = 1
    # 在每一行上找到前k个最小值
    values, _ = torch.kthvalue(tensor, k, dim=1)

    # 获取输入张量的行数
    num_rows = tensor.size(0)

    # 扩展percentile_values，使其形状与输入张量一致
    expanded_values = values.view(num_rows, 1).expand(-1, tensor.size(1))

    # 比较每个值与对应的values[i]，并根据比较结果进行填充
    result = torch.where(tensor <= expanded_values, torch.tensor(0.1), torch.tensor(1.))

    return result


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(), normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9*dsize)
    # print(dsize, tr_size, dsize - tr_size)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"],
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"],
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"],
                                      batch_size=train_bs * 3,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output),1)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def KLLoss(input_, target_,min):
    softmax = nn.Softmax(dim=1)(input_)
    kl_loss = (- target_ * torch.log(softmax + 1e-6)).sum(dim=1)
    kl_loss = torch.where(min == args.batch_size+1, args.sigma * kl_loss, kl_loss)
    return kl_loss.mean(dim=0)
def mixup(x, t_batch, netF, netB, netC, min,args):
    lam = (torch.from_numpy(np.random.beta(1., 1., [len(x)]))).float().cuda()
    # t_batch = torch.eye(args.class_num)[t_batch.argmax(dim=1)].cuda() # onehot
    shuffle_idx = torch.randperm(len(x))

    mixed_x = (lam * x.permute(1, 2, 3, 0) + (1 - lam) * x[shuffle_idx].permute(1, 2, 3, 0)).permute(3, 0, 1, 2)  # 混合的x
    mixed_t = (lam.unsqueeze(1) * t_batch + (1 - lam.unsqueeze(1)) * t_batch[shuffle_idx])  # 混合的标签

    for i in range(t_batch.size(0)):
        if min[i]==args.batch_size+2:
            mixed_x[i] = x[i]
            mixed_t[i] = t_batch[i]

    mixed_x, mixed_t = map(torch.autograd.Variable, (mixed_x, mixed_t))
    mixed_outputs = netC(netB(netF(mixed_x)))

    return KLLoss(mixed_outputs, mixed_t,min)


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    crop_size = 224
    augment1 = transforms.Compose([
        # transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
    ])

    param_group = []
    param_group_c=[]
    for k, v in netF.named_parameters():
        #if k.find('bn')!=-1:
        if True:
            param_group += [{'params': v, 'lr': args.lr * 0.1}]

    for k, v in netB.named_parameters():
        if True:
            param_group += [{'params': v, 'lr': args.lr * 1}]
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)


    #building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample=len(loader.dataset)
    fea_bank=torch.randn(num_sample,256)
    score_bank = torch.randn(num_sample, 12).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx=data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm=F.normalize(output)
            outputs = netC(output)
            outputs=nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    acc_log=0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        alpha = (1 + 10 * float(iter_num) / float(max_iter)) ** (-args.beta)
        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        #output_re = softmax_out.unsqueeze(1)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_@fea_bank.T
            _, idx_near = torch.topk(distance,
                                    dim=-1,
                                    largest=True,
                                    k=args.K+1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]    #batch x K x C

            fea_near = fea_bank[idx_near]  #batch x K x num_dim


            expanded_output_f_ = output_f_.unsqueeze(1).expand(-1, args.K,-1)  # batch x K x C
            distance_p = torch.norm(expanded_output_f_.cuda() - fea_near.cuda(), p=2, dim=2)
            normalized_weights = distance_p / torch.sum(distance_p, dim=1, keepdim=True)

            weight_pse = Entropy(score_near, 2)  # bs*k
            weight_pse = 1. / weight_pse
            normalized_entro = weight_pse / torch.sum(weight_pse, dim=1, keepdim=True)


            weighted_score = torch.einsum('bkc,bk->bc', score_near, normalized_entro/normalized_weights)
            weighted_score = nn.Softmax(dim=1)(weighted_score)
            max_values, max_indices = torch.max(weighted_score, dim=1)
            p_onehot = torch.eye(args.class_num)[max_indices]
            distance_o = torch.cdist(output_f_, output_f_, p=2)  # 64 64
            mask2 = fill_tensor_based_on_percentile(distance_o, args.r)  # r

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K,
                                                         -1)  # batch x K x C
        l2 = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1).cuda()).sum(1))


        #MPM
        bs, k, c = score_near.size()
        min = torch.argmin(Entropy(score_near,2), dim=1)
        min_ = min.view(bs, 1, 1)
        selected = torch.gather(score_near, 1, min_.expand(bs, 1, c))
        selected = selected.squeeze(1)
        tf = selected.detach().clone()
        tf_ = softmax_out.detach().clone()
        min = min.detach()
        tf,flag = newre(tf, tf_, min,p_onehot)  # soft

        #
        images1 = torch.autograd.Variable(augment1(inputs_test))
        images1 = images1.cuda()
        pred = torch.autograd.Variable(tf)
        l6 = mixup(images1, pred, netF, netB, netC, min, args)


        # data sep
        copy = softmax_out.T  # .detach().clone()#
        dot_neg = softmax_out @ copy  # batch x batch
        dot_neg = (dot_neg * mask2.T.cuda()).sum(-1)  # batch


        neg_pred = torch.mean(dot_neg)
        l4 = neg_pred*alpha


        loss=l2+l4+l6#+l3

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()


        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == 'visda-2017':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB,
                                             netC,flag= True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(
                    args.name, iter_num, max_iter, acc_s_te
                ) + '\n' + 'T: ' + acc_list

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()

            if acc_s_te>acc_log:
                acc_log=acc_s_te
                torch.save(
                    netF.state_dict(),
                    osp.join(args.output_dir, "target_F_" + '2021_'+str(args.K) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(args.output_dir,
                                "target_B_" + '2021_' + str(args.K) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(args.output_dir,
                                "target_C_" + '2021_' + str(args.K) + ".pt"))


    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='6',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=30,
                        help="max iterations")
    parser.add_argument('--interval', type=int, default=150)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument(
        '--dset',
        type=str,
        default='visda-2017')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net',
                        type=str,
                        default='resnet101')
    parser.add_argument('--seed', type=int, default=2023, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--KK', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('-kl', type=float, default=0.2)
    parser.add_argument('-sigma', type=float, default=0.5)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='weight/target/')
    parser.add_argument('--output_src', type=str, default='weight/source/')
    parser.add_argument('--tag', type=str, default='selfplus')
    parser.add_argument('--da',
                        type=str,
                        default='uda')
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument("-beta", type=float, default=5.0)
    parser.add_argument('-r', type=float, help='r',
                        default=0.1)
    parser.add_argument('-log',
                        type=str,
                        help='directory of log',
                        default='tar')

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/train/' + names[
            args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/validation/' + names[
            args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/validation/' + names[
            args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset,
                                       names[args.s][0].upper())
        args.output_dir = osp.join(
            args.output, args.da, args.dset,
            names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(
            osp.join(args.output_dir, args.log+'.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)
