import torch
from torch import nn

def get_loss_function(loss_name,weights = None, reduction="mean"):
    if weights is not None:
        weights=torch.FloatTensor(weights)
    if loss_name.lower() == 'weightedcategoricalcrossentropy':
        return WeightedCategoricalCrossEntropy(weights=weights,reduction=reduction)
    else:
        raise ValueError('Loss name %s not recognized' % loss_name)


class WeightedCategoricalCrossEntropy(nn.Module):
    def __init__(self,weights=None, reduction = 'mean'):
        super(WeightedCategoricalCrossEntropy, self).__init__()
        if weights is not None:
            self.weights = weights.unsqueeze(-1).unsqueeze(-1)
        else:
            self.weights = None
        self.reduction = reduction
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, y_pred,y_true):
        predictions = nn.functional.softmax (y_pred,dim=1)
        predictions = torch.clip(predictions,self.eps , 1 - self.eps)
        loss = -y_true * torch.log(predictions) 
        if self.weights is not None:
            loss *= self.weights.to(y_pred)   
        
        if self.reduction =='mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

