from __future__ import print_function
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import torchvision.transforms as transforms
import os
import argparse
import evaluation
from fusedataloader import Imagedata
import model as models
from LPLModelbuilder import Network
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation

dict = models.__dict__
model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MMOSR Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--arch', default='OpenVGG13', choices=model_names, type=str, help='choosing network')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--es', default=100, type=int, help='epoch size')
parser.add_argument('--train_class_num', default=9, type=int, help='Classes used in training')
parser.add_argument('--evaluate', default=False, action='store_true', help='Evaluate without training')
parser.add_argument('--alpha', default = 5, type = int, help='Magnitude of the anchor center')
parser.add_argument('--var_threshold', default = 0.4, type = float, help='variance threshold')
parser.add_argument('--lbda', default = 0.1, type = float, help='Weighting of Distance loss')
parser.add_argument('--gpu', default = '0', type=str, help='GPU iDX')

args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(0)
    start_epoch = 0  

    # checkpoint
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print("Model name: ", args.arch)
    print("Traing Data trial: ", args.trial)
    transform_train = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
    
    trainset = Imagedata(image_path='./LMTData/Train'+str(args.trial)+'/Image',spec_path='./LMTData/Train'+str(args.trial)+'/Spectrogram', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
    
    testset = Imagedata(image_path='./LMTData/Test'+str(args.trial)+'/Image',spec_path='./LMTData/Test'+str(args.trial)+'/Spectrogram', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)

    # Model
    print('==> Building model..')
    net = Network(backbone=args.arch, num_classes=args.train_class_num,embed_dim=None)
    net = net.to(device)
    img_anchors = torch.diag(torch.Tensor([args.alpha for i in range(args.train_class_num)]))
    spec_anchors = torch.diag(torch.Tensor([args.alpha for i in range(args.train_class_num)]))
    anchors = torch.diag(torch.Tensor([args.alpha for i in range(args.train_class_num)]))
    net.set_anchors(img_anchors, spec_anchors, anchors)
    criterion_softmax = nn.CrossEntropyLoss()
    optimizer_model = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'Total Loss', 'train Acc.'])

    #epoch=0
    if not args.evaluate:
        for epoch in range(start_epoch, args.es):
            print('\nEpoch: %d   Learning rate: %f' % (epoch+1, optimizer_model.param_groups[0]['lr']))
            adjust_learning_rate(optimizer_model, epoch, args.lr)
            train_loss, train_acc = train(net,trainloader,optimizer_model, criterion_softmax, device)
            logger.append([epoch+1, train_loss, train_acc])
    save_model(net, epoch, os.path.join(args.checkpoint,'LMT_%d.pth'% (args.trial)))
    test(net, trainloader, testloader, args.var_threshold, device)
    logger.close()

def JointLoss(distances, gt):
    true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1) 
    non_gt = torch.Tensor([[i for i in range(args.train_class_num) if gt[x] != i] for x in range(len(distances))]).long().cuda()
    others = torch.gather(distances, 1, non_gt) 
    anchor = torch.mean(true)
    tuplet = torch.exp(-others+true.unsqueeze(1))
    tuplet = torch.mean(torch.log(1+torch.sum(tuplet, dim = 1)))
    total = args.lbda*anchor + tuplet
    return total, anchor, tuplet

# Training
def train(net, trainloader, optimizer_model, criterion_softamx, device):
    net.train()
    totoal_loss = 0
    correct = 0
    total = 0
    for batch_idx, (img, spec, targets) in enumerate(trainloader):
        img, spec, targets = img.to(device), spec.to(device), targets.to(device)
        _, _, _, outDistance1, outDistance2, outDistance = net(img, spec)
        _, _, tupletLoss = JointLoss(outDistance, targets)
        _, anchorLoss1, _ = JointLoss(outDistance1, targets)
        _, anchorLoss2, _ = JointLoss(outDistance2, targets)
        loss = tupletLoss + args.lbda*(anchorLoss1+anchorLoss2)
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        totoal_loss += loss.item()
        _, predicted = outDistance.min(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader),'Loss:%.3f  Acc: %.3f%% (%d/%d)'% (totoal_loss/(batch_idx+1), 100.*correct/total,correct, total))
    return totoal_loss/(batch_idx+1), correct/total

def test(net, trainloader, testloader, var_threshold, device):
    net.eval()
    print("Update the class centers...")
    anchor_means, img_means, spec_means= find_anchor_means(net, trainloader)
    net.set_anchors(torch.Tensor(img_means), torch.Tensor(spec_means), torch.Tensor(anchor_means))
    radius, radius_img, radius_spec = find_radius(net, trainloader)
    net.set_radius(torch.Tensor(radius), torch.Tensor(radius_img), torch.Tensor(radius_spec))
    scores, labels, relia_v= [], [], []
    print('==> Evaluating MM open set network')
    with torch.no_grad():
        for batch_idx,(img, spec, targets) in enumerate(testloader):
            img, spec, targets = img.to(device), spec.to(device), targets.to(device)
            _, _,_ , _, _, outDistance, relia_var = net(img, spec, train=False)
            scores.append(-outDistance)
            labels.append(targets)
            relia_v.append(relia_var)
            progress_bar(batch_idx, len(testloader))
    # Get the prdict results.  
    labels = torch.cat(labels,dim=0)
    var = torch.cat(relia_v,dim=0)
    scores = torch.cat(scores,dim=0)
    
    mask = var < var_threshold
    labels = labels[mask]
    scores = scores[mask]
    print(labels.shape)

    labels = labels.cpu().numpy() 
    scores = scores.softmax(dim=1).cpu().numpy()
    
    pred_known = scores[np.where(labels<args.train_class_num)]
    label_knonwn = labels[np.where(labels<args.train_class_num)]
    pred_unknown = scores[np.where(labels==args.train_class_num)]
    
    x1, x2 = np.max(pred_known, axis=1), np.max(pred_unknown, axis=1)
    pred_k = np.argmax(pred_known, axis=1)
    close_acc = ((pred_k == label_knonwn).sum()/label_knonwn.shape[0])*100
    results = evaluation.metric_ood(x1, x2)['Bas']   
    oscr_socre = evaluation.compute_oscr(pred_known, pred_unknown, label_knonwn) 
    results['OSCR'] = oscr_socre * 100.
    results['CloseACC'] = close_acc
    
    print(f"LPL AUROC is %.3f" % (results['AUROC']))
    print(f"LPL OSCR is %.3f" % (results['OSCR']))
    print(f"LPL Close_acc is %.3f" % (results['CloseACC']))

def save_model(net, epoch, path):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, path)

def find_anchor_means(net, trainloader):
    #find gaussians for each class
    device = 'cuda'
    logits, logits1, logits2=[], [], []
    labels=[]
    for _, (img, spec, targets) in enumerate(trainloader):
        img, spec, targets = img.to(device), spec.to(device), targets.to(device)
        logit1, logit2, logit, _, _, _ = net(img, spec)  

        logits += logit.cpu().detach().tolist()
        logits1 += logit1.cpu().detach().tolist()
        logits2 += logit2.cpu().detach().tolist()
        labels += targets.cpu().tolist()  
        
    logits = np.asarray(logits)
    logits1= np.asarray(logits1)
    logits2 = np.asarray(logits2)
    labels = np.asarray(labels)
    num_classes = args.train_class_num
    means_img = [None for i in range(num_classes)]
    means_spec = [None for i in range(num_classes)]
    means = [None for i in range(num_classes)]
    for cl in range(num_classes):
        #fuse center
        x = logits[labels == cl]
        x = np.squeeze(x)
        means[cl] = np.mean(x, axis = 0)
        #img center
        x_img = logits1[labels == cl]
        x_img = np.squeeze(x_img)
        means_img[cl] = np.mean(x_img, axis = 0)
        #spec center
        x_spec = logits2[labels == cl]
        x_spec = np.squeeze(x_spec)
        means_spec[cl] = np.mean(x_spec, axis = 0)
    return means, means_img, means_spec

def find_radius(net, trainloader):
    #find gaussians for each class
    device = 'cuda'
    distance, distance1, distance2=[], [], []
    labels=[]
    for _, (img, spec, targets) in enumerate(trainloader):
        img, spec, targets = img.to(device), spec.to(device), targets.to(device)
        _, _, _, outDistance1, outDistance2, outDistance = net(img, spec)
        outDistance = torch.gather(outDistance, 1, targets.view(-1, 1)).view(-1)
        outDistance1 = torch.gather(outDistance1, 1, targets.view(-1, 1)).view(-1)
        outDistance2 = torch.gather(outDistance2, 1, targets.view(-1, 1)).view(-1)
        distance += outDistance.cpu().detach().tolist()
        distance1 += outDistance1.cpu().detach().tolist()
        distance2 += outDistance2.cpu().detach().tolist()
        labels += targets.cpu().tolist()      
    distance = np.asarray(distance)
    distance1= np.asarray(distance1)
    distance2 = np.asarray(distance2)
    labels = np.asarray(labels)
    distance = distance[:,np.newaxis]
    distance1 = distance1[:,np.newaxis]
    distance2 = distance2[:,np.newaxis]
    radius_img = [None for i in range(args.train_class_num)]
    radius_spec = [None for i in range(args.train_class_num)]
    radius = [None for i in range(args.train_class_num)]
    for cl in range(args.train_class_num):
        #fuse center
        x = distance[labels == cl]
        x = np.squeeze(x)
        radius[cl] = np.mean(x, axis = 0)
        #img center
        x_img = distance1[labels == cl]
        x_img = np.squeeze(x_img)
        radius_img[cl] = np.mean(x_img, axis = 0)
        #spec center
        x_spec = distance2[labels == cl]
        x_spec = np.squeeze(x_spec)
        radius_spec[cl] = np.mean(x_spec, axis = 0)
    return radius, radius_img, radius_spec

if __name__ == '__main__':
    args = parser.parse_args()
    args.checkpoint = './checkpoints/LPL/' + args.arch
    for i in range (1,4):
        args.trial = i
        main()