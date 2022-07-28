import torch

@torch.no_grad()
def val(model, dataloader, device, compute_loss):

    # 测试步骤开始
    pred = []
    gt = []
    total_loss = 0
    
    model.eval()
    for data in dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        outputs = model(imgs)
        # 整个batch的损失
        test_loss = compute_loss(outputs, targets)
        total_loss += test_loss.item()

        pred.append(outputs.argmax(1))
        gt.append(targets)

    mean_loss = total_loss / len(dataloader)
        
    pred = torch.cat(pred, dim=0)
    gt = torch.cat(gt, dim=0)
    
    # 计算整体的准确率
    total_num = gt.shape[0]
    total_num_TPs = (pred == gt).sum().item()
    total_accuracy = total_num_TPs / total_num

    # 逐个类别分别计算准确率
    accuracy_classes = {}
    classes_name = model.classes_name
    for class_name in classes_name:
        cls_id = classes_name.index(class_name)

        idx = gt == cls_id
        ## 取出该类别的gt
        gt_class = gt[idx]
        pred_class = pred[idx]

        num_class = gt_class.shape[0]
        num_class_TPs = (pred_class == gt_class).sum().item()

        accuracy_class = num_class_TPs / num_class
        accuracy_classes[class_name] = accuracy_class
    

    return mean_loss, total_accuracy, accuracy_classes



