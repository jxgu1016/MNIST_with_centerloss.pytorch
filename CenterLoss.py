import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import Function


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim ):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.reg_weight = reg_weight

    def forward(self, label, feature):
        return self.centerlossfunc(feature, label, self.centers)


class CenterlossFunc(Function):

    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum(1).sum(0) / 2.0


    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_variables
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        center_grads = Variable(torch.zeros(centers.size()))
        center_grads.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        return grad_output*diff, None, center_grads

def main(test_cuda=False):
    print('-'*80)
    ct = CenterLoss(10,2)
    y = Variable(torch.Tensor([0,0,2,1]))
    feat = Variable(torch.zeros(4,2),requires_grad=True)
    if test_cuda:
        ct = ct.cuda()
        y = Variable(torch.Tensor([0,0,2,1]).cuda())
        feat = Variable(torch.zeros(4,2).cuda(),requires_grad=True)
    print (list(ct.parameters()))
    print (ct.centers.grad)
    # print y
    # print feat
    out = ct(y,feat)
    out.backward()
    print(ct.centers.grad)
    print (feat.grad)


if __name__ == '__main__':
    torch.manual_seed(999)
    main(test_cuda=False)
    if torch.cuda.is_available():
        main(test_cuda=True)
