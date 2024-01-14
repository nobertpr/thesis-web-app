import torch
import lightning.pytorch as pl
import torchmetrics
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DeepLabV3Plus_pl(pl.LightningModule):
    def __init__(self, n_classes, lr=1e-4, encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()
        self.n_classes = n_classes

        
        self.model = smp.DeepLabV3Plus(encoder_name=encoder_name,encoder_weights=encoder_weights,classes=n_classes)
        self.lr = lr
        self.save_hyperparameters('n_classes','encoder_name','encoder_weights','lr')
        
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, average='macro')
        self.accuracy_class = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, average=None)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=n_classes, average='macro')
        self.precision_class = torchmetrics.Precision(task="multiclass", num_classes=n_classes, average=None)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=n_classes, average='macro')
        self.recall_class = torchmetrics.Recall(task="multiclass", num_classes=n_classes, average=None)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average='macro')
        self.f1_class = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average=None)
        self.jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=n_classes, average='macro')
        self.jaccard_class = torchmetrics.JaccardIndex(task="multiclass", num_classes=n_classes,average=None)
        self.dice = torchmetrics.Dice(num_classes=n_classes, average='macro')
#         self.dice_class = torchmetrics.Dice(num_classes=n_classes, average=None)
        
        # Class Weights
        self.class_weights = [0.00868963, 0.58788445, 0.01520088, 0.07803337, 0.31019167]
        self.class_weights_unnorm = torch.tensor([1.7258057494713657, 116.75693945538318, 3.018974415508092, 15.497838011613178, 61.60569406906407], dtype=torch.float32)
        
#         self.loss = smp.losses.JaccardLoss(mode='multiclass')
#         self.loss = smp.losses.DiceLoss(mode='multiclass')
        self.loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.0)
        
#         self.loss = focal_loss_utils.WFocalLoss(mode='multiclass', gamma=2.0, alpha=self.class_weights)
#         self.loss = torch.hub.load(
#             'adeelh/pytorch-multi-class-focal-loss',
#             model='FocalLoss',
#             alpha=torch.tensor(self.class_weights),
#             gamma=2,
#             reduction='mean',
#         )
        
#         self.loss = nn.CrossEntropyLoss()
        
#         self.loss = nn.CrossEntropyLoss(weight=self.class_weights_unnorm)
        
        self.id2label = {
            0:"Background",
            1:"Building",
            2:"Woodland",
            3:"Water",
            4:"Road",
        }

    def forward(self, x):
        logits = self.model(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        
        y_hat_argmax = torch.argmax(torch.softmax(y_hat,dim=1),dim=1)
        loss = self.loss(y_hat, y)
        
        # Torch metric
        tm_acc = self.accuracy(y_hat_argmax,y)
        tm_rec = self.recall(y_hat_argmax,y)
        tm_prec = self.precision(y_hat_argmax,y)
        tm_iou = self.jaccard(y_hat_argmax,y)
        tm_f1 = self.f1(y_hat_argmax,y)
        tm_dice = self.dice(y_hat_argmax,y)
        
        
        self.log_dict({
            "train_loss":loss,
            "train_acc":tm_acc,
            "train_iou":tm_iou,
            "train_f1":tm_f1,
            "train_precision":tm_prec,
            "train_recall":tm_rec,
            "train_dice":tm_dice,
        },prog_bar=True, on_step=True, on_epoch=True)
        
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        
        
        y_hat_argmax = torch.argmax(torch.softmax(y_hat,dim=1),dim=1)
        loss = self.loss(y_hat, y)
        
        # Torch metric
        tm_acc = self.accuracy(y_hat_argmax,y)
        tm_acc_class = self.accuracy_class(y_hat_argmax,y)
        tm_rec = self.recall(y_hat_argmax,y)
        tm_rec_class = self.recall_class(y_hat_argmax,y)
        tm_prec = self.precision(y_hat_argmax,y)
        tm_prec_class = self.precision_class(y_hat_argmax,y)
        tm_iou = self.jaccard(y_hat_argmax,y)
        tm_iou_class = self.jaccard_class(y_hat_argmax,y) 
        tm_f1 = self.f1(y_hat_argmax,y)
        tm_f1_class = self.f1_class(y_hat_argmax,y)
        tm_dice = self.dice(y_hat_argmax,y)
#         tm_dice_class = self.dice_class(y_hat_argmax,y)
        
        log_dict = {
            "val_loss":loss,
            "val_acc":tm_acc,
            "val_iou":tm_iou,
            "val_f1":tm_f1,
            "val_precision":tm_prec,
            "val_recall":tm_rec,
            "val_dice":tm_dice
        }
        
        for i, x in enumerate(tm_acc_class):
            log_dict[f'val_acc_class_{self.id2label[i]}'] = x
        
        for i, x in enumerate(tm_iou_class):
            log_dict[f'val_iou_class_{self.id2label[i]}'] = x
            
        for i, x in enumerate(tm_f1_class):
            log_dict[f'val_f1_class_{self.id2label[i]}'] = x
            
        for i, x in enumerate(tm_prec_class):
            log_dict[f'val_precision_class_{self.id2label[i]}'] = x
            
        for i, x in enumerate(tm_rec_class):
            log_dict[f'val_recall_class_{self.id2label[i]}'] = x
            
        
        self.log_dict(
            log_dict
        ,prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)