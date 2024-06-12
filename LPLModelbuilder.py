import sys
sys.path.append("../..")
import model as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy

class Network(nn.Module):
    def __init__(self, backbone='ResNet18', num_classes=1000,embed_dim=None):
        super(Network, self).__init__()
        self.backbone_name = backbone
        self.backbone = models.__dict__[backbone](num_classes=num_classes,backbone_fc=False)
        self.dim = self.get_backbone_last_layer_out_channel()
        self.num_classes = num_classes
        self.img_anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes), requires_grad = False)
        self.spec_anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes), requires_grad = False)
        self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes), requires_grad = False)
        
        self.radius_img = nn.Parameter(torch.zeros(self.num_classes), requires_grad = False)
        self.radius_spec = nn.Parameter(torch.zeros(self.num_classes), requires_grad = False)
        self.radius = nn.Parameter(torch.zeros(self.num_classes), requires_grad = False)
        
        self.classifier1 = nn.Linear(self.dim, num_classes)
        self.classifier2 = nn.Linear(self.dim, num_classes)

    def get_backbone_last_layer_out_channel(self):
        if self.backbone_name[-5:] == 'VGG13':
            return 512
        else:
            last_layer = list(self.backbone.children())[-1]
        if isinstance(last_layer, nn.BatchNorm2d):
            return last_layer.num_features
        else:
            return last_layer.out_channels
  
    def forward(self,img, spec, train = True):
        batch_size = img.size(0)
        feature1,feature2 = self.backbone(img,spec)
        feature1 = F.adaptive_avg_pool2d(feature1,1)
        feature2 = F.adaptive_avg_pool2d(feature2,1)
        feature1 = feature1.view(img.size(0), -1)
        feature2 = feature2.view(spec.size(0), -1)
        logits1 = self.classifier1(feature1)#特征->逻辑向量 
        logits2 = self.classifier2(feature2)
        outDistance1 = self.cos_classifier(logits1,self.img_anchors)
        outDistance2 = self.cos_classifier(logits2,self.spec_anchors)
        Distance1 = self.distance_classifier(logits1,self.img_anchors)
        Distance2 = self.distance_classifier(logits2,self.spec_anchors)
        entropy1 = entropy(F.normalize(1/Distance1,p=1,dim=1).cpu().detach().numpy(),axis=1)
        entropy2 = entropy(F.normalize(1/Distance2,p=1,dim=1).cpu().detach().numpy(),axis=1)
        entropy1=torch.exp(-torch.from_numpy(entropy1))
        entropy2=torch.exp(-torch.from_numpy(entropy2))
        entropy_sum=entropy1+entropy2
        entropy_weight1=entropy1/entropy_sum
        entropy_weight1=entropy_weight1.unsqueeze(1).expand(-1,self.num_classes).cuda()
        entropy_weight2=entropy2/entropy_sum
        entropy_weight2=entropy_weight2.unsqueeze(1).expand(-1,self.num_classes).cuda()
        logits1_weighted = torch.mul(logits1,entropy_weight1)#img
        logits2_weighted = torch.mul(logits2,entropy_weight2)#spec
        logits=logits1_weighted+logits2_weighted

        outDistance = self.cos_classifier(logits,self.anchors)
        
        if train == False:
            outDistance = self.distance_classifier(logits,self.anchors)
            outDistance1 = self.distance_classifier(logits1,self.img_anchors)
            outDistance2 = self.distance_classifier(logits2,self.spec_anchors)
            _,index_center = torch.sort(outDistance)
            nearest_index = index_center[:,0]
            radius_img = self.radius_img.expand(batch_size, self.num_classes)
            radius_spec = self.radius_spec.expand(batch_size, self.num_classes)
            relia_img = radius_img.gather(1,nearest_index.view(-1, 1))/outDistance1.gather(1,nearest_index.view(-1, 1))
            relia_spec = radius_spec.gather(1,nearest_index.view(-1, 1))/outDistance2.gather(1,nearest_index.view(-1, 1))
            relia_mm = torch.cat((relia_img,relia_spec),1)
            relia_mm = F.normalize(relia_mm,p=1,dim=1)
            relia_var = torch.var(relia_mm,dim=1)
            reliability = relia_img*relia_spec
            outDistance_calibrate = outDistance*reliability
            return logits1,logits2,logits,outDistance1,outDistance2,outDistance_calibrate,relia_var
        
        return logits1,logits2,logits,outDistance1,outDistance2,outDistance

    def cos_classifier(self, x, centers):
        n = x.size(0)
        m = self.num_classes
        d = self.num_classes
        x = x.unsqueeze(1).expand(n, m, d)
        anchors = centers.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x-anchors, 2, 2)
        cos = nn.functional.cosine_similarity(x,anchors, 2)
        dists = dists+(1-cos)
        return dists
    
    def distance_classifier(self, x, centers):
        n = x.size(0)
        m = self.num_classes
        d = self.num_classes
        x = x.unsqueeze(1).expand(n, m, d)
        anchors = centers.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x-anchors, 2, 2)
        return dists
    
    def set_anchors(self, img_means, spec_means, anchor_means):
        self.img_anchors = nn.Parameter(img_means, requires_grad = False)
        self.spec_anchors = nn.Parameter(spec_means, requires_grad = False)
        self.anchors = nn.Parameter(anchor_means, requires_grad = False)
        self.cuda()
        
    def set_radius(self, radius, radius_img, radius_spec):
        self.radius_img = nn.Parameter(radius_img, requires_grad = False)
        self.radius_spec = nn.Parameter(radius_spec, requires_grad = False)
        self.radius = nn.Parameter(radius, requires_grad = False)
        self.cuda()