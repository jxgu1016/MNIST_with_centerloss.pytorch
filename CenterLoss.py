import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import Function


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim ):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        return self.centerlossfunc(feat, label, self.centers)


class CenterlossFunc(Function):

    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum(1).sum(0) / 2.0


    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new(centers.size(0)).fill_(1)
        ones = centers.new(label.size(0)).fill_(1)
        grad_centers = centers.new(centers.size()).fill_(0)
        counts = counts.scatter_add_(0, label.long(), ones)
        # print counts, grad_centers
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)


        grad_centers = grad_centers/counts.view(-1, 1)

        return Variable(-grad_output.data*diff), None, Variable(grad_centers)

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
