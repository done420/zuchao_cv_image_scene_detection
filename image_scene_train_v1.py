##coding: utf-8

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import pdb
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=365, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='places365',help='which dataset to train')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    # create model
    print(("=> creating model '{}'".format(args.arch)))

    model = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.arch.lower().startswith('alexnet') or args.arch.lower().startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    print(model)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),#transforms.RandomSizedCrop(224)
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),#transforms.Scale(256)
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.arch.lower())


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()#target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0)) #losses.update(loss.data[0], input.size(0))
        top1.update(prec1.item(), input.size(0))#top1.update(prec1[0], input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5)))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()#target.cuda(async=True)
        input_var = torch.autograd.Variable(input) #torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target) #torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))#losses.update(loss.data[0], input.size(0))
        top1.update(prec1.item(), input.size(0))#top1.update(prec1[0], input.size(0))
        top5.update(prec5.item(), input.size(0))#top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))

    print((' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5)))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# if __name__ == '__main__':
#     main()


CLASS_NAME = {
    "gazebo":"露台",
    "balcony":"阳台",
    "corridor":" 走廊",
    "courtyard":"院子",
    "yard":"院子",
    "tree_house":"树屋",
    "formal_garden":"正式花园",
    "japanese_garden":"日式花园",
    "roof_garden":" 屋顶花园",
    "topiary_garden":"修剪花园",
    "zen_garden":"禅意花园",
    "swimming_pool":"游泳池",
    "porch":"玄关",
    "atrium":"中庭",
    "living_room":"客厅",
    "dining_room":"餐厅",
    "kitchen":"厨房",
    "bathroom":"浴室",
    "shower":"淋浴房",
    "bedroom":"卧室",
    "childs_room":"儿童房",
    "tea_room":"茶室",
    "home_office":"家庭工作室",
    "closet":"衣帽间",
    "dressing_room":"衣帽间",
    "home_theater":"影音室",
    "television_room":"电视房",
    "recreation_room":"娱乐室",
    "staircase":"楼梯",
    "basement":"地下室",
    "storage_room":"储藏室",
    "pantry":"储藏室",
    "garage":"车库"
}


CLASS_NAME2 = {
    "gazebo-exterior":"露台",
    "balcony-exterior":"阳台_外景",
    "balcony-interior":"阳台_内景",
    "corridor":" 走廊",
    "courtyard":"院子",
    "yard":"院子",
    "tree_house":"树屋",
    "formal_garden":"正式花园",
    "japanese_garden":"日式花园",
    "roof_garden":" 屋顶花园",
    "topiary_garden":"修剪花园",
    "zen_garden":"禅意花园",
    "swimming_pool-indoor":"游泳池_室内",
    "swimming_pool-outdoor":"游泳池_室外",
    "porch":"玄关",
    "atrium-public":"中庭",
    "living_room":"客厅",
    "dining_room":"餐厅",
    "kitchen":"厨房",
    "bathroom":"浴室",
    "shower":"淋浴房",
    "bedroom":"卧室",
    "childs_room":"儿童房",
    "tea_room":"茶室",
    "home_office":"家庭工作室",
    "closet":"衣帽间",
    "dressing_room":"衣帽间",
    "home_theater":"影音室",
    "television_room":"电视房",
    "recreation_room":"娱乐室",
    "staircase":"楼梯",
    "basement":"地下室",
    "storage_room":"储藏室",
    "pantry":"储藏室",
    "garage-indoor":"车库_室内",
    "garage-outdoor":"车库_室外"
}



class_name = {
    'atrium-public': 0,
    'balcony-exterior': 1,
    'balcony-interior': 2,
    'basement': 3,
    'bathroom': 4,
    'bedroom': 5,
    'childs_room': 6,
    'closet': 7,
    'corridor': 8,
    'courtyard': 9,
    'dining_room': 10,
    'dressing_room': 11,
    'formal_garden': 12,
    'garage-indoor': 13,
    'garage-outdoor': 14,
    'gazebo-exterior': 15,
    'home_office': 16,
    'home_theater': 17,
    'japanese_garden': 18,
    'kitchen': 19,
    'living_room': 20,
    'pantry': 21,
    'porch': 22,
    'recreation_room': 23,
    'roof_garden': 24,
    'shower': 25,
    'staircase': 26,
    'storage_room': 27,
    'swimming_pool-indoor': 28,
    'swimming_pool-outdoor': 29,
    'tea_room': 30,
    'television_room': 31,
    'topiary_garden': 32,
    'tree_house': 33,
    'yard': 34,
    'zen_garden': 35
}



a = sorted(class_name.items(), key=lambda item:item[1])

print([ele[0] for ele in a])


# input_root = "/home/user/qunosen/2_project/4_train/4_places/2_my_set_new/all"
# save_root = "/home/user/qunosen/2_project/4_train/4_places/2_my_set_new/select/"
#
#
#
# def get_dat(input_root,save_root):
#     import shutil
#     import glob,random
#     from tqdm import tqdm
#     num = 250
#     for cat in os.listdir(input_root):
#         cat_path = os.path.join(input_root,cat)
#         imgs = glob.glob(cat_path + "/*")
#         save_dir = os.path.join(save_root,cat)
#         if not os.path.exists(save_dir):os.makedirs(save_dir)
#
#         if len(imgs)> num:
#             imgs_to_use = [i for i in random.sample(imgs, num)]
#
#             for file_name in tqdm(imgs_to_use):
#                 save_file = os.path.join(save_dir,os.path.basename(file_name))
#                 shutil.copy(file_name, save_file)
#
#         else:
#             save_path = os.path.join(save_root,cat)
#             imgs = glob.glob(cat_path + "/*")
#
#             for file_name in tqdm(imgs):
#                 save_file = os.path.join(save_dir,os.path.basename(file_name))
#                 shutil.copy(file_name, save_file)
#
#
#
# # get_dat(input_root, save_root)
#
# def check_images(input_root):
#     img_types = [".jpg",'.png','.bmp']
#     a = {}
#     for cat in os.listdir(input_root):
#         cat_path = os.path.join(input_root,cat)
#         num = 0
#         for file_name in os.listdir(cat_path):
#             type = os.path.splitext(file_name)[1]
#             file_path = os.path.join(cat_path,file_name)
#
#             num+=1
#
#             if type not in img_types:
#                 print(file_path)
#         a[cat] = num
#
#     for ele in a:
#         print("{}, count:{}".format(ele,a.get(ele)))
#
# check_images(r'/home/user/qunosen/2_project/4_train/4_places/places365-master/data/image_36_classes/val')
#
#
# import random
#
# ############## 将数据分为训练集和测试集
# def random_split_imglist(input_root,save_root,split_rate = 0.1): #### split val json file from all json
#
#     # split_rate = 0.1
#
#     for cat in os.listdir(input_root):
#         cat_path = os.path.join(input_root,cat)
#         imgfiles_list = [os.path.join(cat_path,file_name) for file_name in os.listdir(cat_path) if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.bmp')]
#         split_num = int(len(imgfiles_list) *split_rate) if int(len(imgfiles_list) *split_rate)>=1 else 1
#
#
#         val_list = [i for i in random.sample(imgfiles_list, split_num)]
#         trian_list = [i for i in imgfiles_list if i not in val_list]
#
#
#         val_dir = os.path.join(save_root,"val",cat)
#         train_dir = os.path.join(save_root,"train",cat)
#         if not os.path.exists(val_dir):os.makedirs(val_dir)
#         if not os.path.exists(train_dir):os.makedirs(train_dir)
#
#         print("from %d split %d json-files for val ..." % (len(imgfiles_list),split_num))
#
#         train_save_list = []
#         val_save_list = []
#         for img_file in val_list:
#             file_name = os.path.basename(img_file)
#             val_file = os.path.join(val_dir,file_name)
#             shutil.copy(img_file,val_file) ###
#             val_save_list.append(file_name)
#             if os.path.exists(val_file):
#                 print("json file_move to %s" % val_file)
#
#         for img_file in trian_list:
#             file_name = os.path.basename(img_file)
#             train_file = os.path.join(train_dir, file_name)
#             shutil.copy(img_file, train_file)  ###
#             train_save_list.append(file_name)
#             if os.path.exists(train_file):
#                 print("json file_move to %s" % train_file)
#
#
# save_dir = r'/home/user/qunosen/2_project/4_train/4_places/places365-master/data/image_36_classes'
# all_json_dir = r'/home/user/qunosen/2_project/4_train/4_places/2_my_set_new/select'
#
# # random_split_imglist(all_json_dir,save_dir)
