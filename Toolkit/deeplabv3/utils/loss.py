import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        #print (c)
        #  class_weight = [0.000302451426998195, 0.033693510058119415 , 0.0008078489860110805, 0.012978597861456307, 0.3089660925571628 , 0.013466730907681955 , 0.30223562383069263 , 0.006694769771890478, 0.0004844173856150665 , 0.0006165261974884125 , 0.159223411143028, 0.15833965590605145, 0.0021903639678039705]
#         class_weight = [0.06, 0.06 , 0.06, 0.06, 0.17 , 0.06 , 0.17 , 0.06, 0.06 , 0.06 , 0.06, 0.06, 0.06]
#         class_weight = [0.05, 0.05 , 0.05, 0.05, 0.225 , 0.05 , 0.225 , 0.05, 0.05 , 0.05 , 0.05, 0.05, 0.05]
#         class_weight = [0.7936124143094874, 0.9144495636572734, 0.8404575025316977, 0.9145104476983651, 1.0, 0.8953173685562668, 0.9975187359232237, 0.8755570686949328, 0.8387191357854562, 0.8614737686919389, 0.912183570586518, 0.9075005010798668, 0.8512908582452005] paper 
#         class_weight = [0.22826921881219303, 0.45122234333673256, 0.2883700900983317, 0.45141512472215756, 1.0, 0.39682683755865206, 0.9686762373267547, 0.35116180948238235, 0.28572855548081927, 0.32358073378700664, 0.444144919215151, 0.4300962498723818, 0.3057237896933413] ln
#         class_weight = [0.05251056694088497, 0.20474876851022586, 0.08464210228085277, 0.2178543394861178, 0.7705885543021966, 0.1652338729965905, 0.7097539662280057, 0.1264295823357223, 0.08442115029737007, 0.10525714639477105, 0.20529009097428866, 0.2003200662999683, 0.09532503988575611] *
        class_weight = [0.4583070978312108, 0.9049870372482407, 0.5818890963586223, 0.9340182267343142, 2.5, 0.8132146857524338, 2.5, 0.7111940764367, 0.5811878250498252, 0.6488694033653857, 0.9063591247430607, 0.8958526157077205, 0.6175249746170682] 
    # + 


        class_weight = torch.tensor (class_weight)
#         print (class_weight.size())
        # self.weight = class_weight
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




