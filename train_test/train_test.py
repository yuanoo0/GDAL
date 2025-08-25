import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import *


def kl_div(source, target, reduction='batchmean'):
    loss = F.kl_div(F.log_softmax(source, 1), target, reduction=reduction)
    return loss


# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores = models['backbone'](inputs)[0]
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

iters = 0
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()
    global iters
    direc = None

    for inputs, labels, _, _ in dataloaders['train']:
        inputs = inputs.cuda()
        labels = labels.cuda()

        # Zero gradients
        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        # Forward pass
        scores, _, _, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        # Compute losses
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        loss = m_backbone_loss + WEIGHT * m_module_loss

        loss.backward()
        # 收集最后一层的梯度
        gradients = [param.grad.flatten() for param in models['backbone'].parameters() if param.grad is not None]
        direc = torch.cat(gradients) if gradients else None

        # Step optimizers
        optimizers['backbone'].step()
        optimizers['module'].step()

    return direc


def train(models, criterion, optimizers, schedulers,  dataloaders, num_epochs, cycle, epoch_loss):
    print('>> Training the model...')
    best_acc = 0.
    direc = None

    for epoch in range(num_epochs):
        direc = train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)
        schedulers['backbone'].step()
        schedulers['module'].step()
        # Save a checkpoint
        if epoch % 20 == 0 or epoch == 199:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
            print('Cycle:', cycle, 'Epoch:', epoch, '---','Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc), flush=True)
    print('>> Finished.')
    return direc

