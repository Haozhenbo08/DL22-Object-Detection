import barlow
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transforms as T
import utils
from engine import train_one_epoch, evaluate
from dataset import UnlabeledDataset, LabeledDataset

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes, pretrained_backbone=True):
    if pretrained_backbone:
        PATH = './backbone/model.pth'
        model = torch.load(PATH)
    else:
        epoch = 1
        model = barlow.getTrainedBarlowModel(epoch)
        
    print('Backbone loaded')

    resnet = model.backbone
    modules = list(resnet.children())[:-2]      # delete the last fc layer.

    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                  aspect_ratios=((0.25,0.33,0.5, 1.0, 2.0,3.0,4.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                      output_size=7,
                                                            sampling_ratio=2)
    model = FasterRCNN(backbone,
                        num_classes=num_classes,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler).cuda()
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)
    
    # Get the model. If pretrained_backbone=True, it will load a pretrained backbone model.pth in the backbone folder.
    # If pretrained_backbone=False, it will train a new backbone with 1 epochs.
    num_classes = 101
    model = get_model(num_classes, pretrained_backbone=False)
    print('FRCNN-R50 loaded.')

    model.to(device)
    model.train()
    
    # Set the bn and conv1 to be non-trainable
    # bn
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)

    # Conv1
    for param in (list(model.children())[1][0]).parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.96, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 1
    print('Labeled traing begins...')
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)
    torch.save(model,'frcnn.pth')
    print("That's it and Hello World!")

if __name__ == "__main__":
    main()
