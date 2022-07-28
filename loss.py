import torch
import torch.nn as nn
import torch.nn.functional as F
    
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true, module=None):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        # if module == 'conf':
        #     # 50不行，35也不行，试一下30/20
        #     index = loss > 30
        #     if torch.any(index):
        #         print(f'pred:{pred[index]}')
        #         print(f'true:{true[index]}')

        #         loss = loss[~index]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    def __init__(self, loss_type:str, num_classes):
        
        self.num_classes = num_classes

        assert loss_type in ('softmax', 'BCE', 'focal_loss')

        self.loss_type = loss_type

        if loss_type == 'softmax':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'BCE':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == 'focal_loss':
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.loss_fn = FocalLoss(self.loss_fn)
        else:
            raise NotImplementedError("loss_type must in ('softmax', 'BCE', 'focal_loss')")



    
    def __call__(self, p, targets):

        if self.loss_type == 'softmax':
            loss = self.loss_fn(p, targets)
        elif self.loss_type in ('BCE', 'focal_loss'):
            # 进行独热编码，并转换数据类型
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).to(p.dtype)
            loss = self.loss_fn(p, targets_one_hot)

        else:
            raise NotImplementedError("loss_type must in ('softmax', 'BCE', 'focal_loss')")
        
        return loss