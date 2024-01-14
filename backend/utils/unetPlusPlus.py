import torch
import lightning.pytorch as pl
import torchmetrics
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class UNetPlusPlus(pl.LightningModule):
    def __init__(self, n_classes, lr=1e-4, encoder_name="resnet50", encoder_weights="imagenet", id2label=None):
        super().__init__()
        self.n_classes = n_classes

        
        self.model = smp.UnetPlusPlus(encoder_name=encoder_name,encoder_weights=encoder_weights,classes=n_classes)
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
        
        if id2label is None:
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
        loss = F.cross_entropy(y_hat, y)
        
        y_hat_argmax = torch.argmax(torch.softmax(y_hat,dim=1),dim=1)
        
        # Torch metric
        tm_acc = self.accuracy(y_hat_argmax,y)
        tm_rec = self.recall(y_hat_argmax,y)
        tm_prec = self.precision(y_hat_argmax,y)
        tm_iou = self.jaccard(y_hat_argmax,y)
        tm_f1 = self.f1(y_hat_argmax,y)
        
        
        self.log_dict({
            "train_loss":loss,
            "train_acc":tm_acc,
            "train_iou":tm_iou,
            "train_f1":tm_f1,
            "train_precision":tm_prec,
            "train_recall":tm_rec,
        },prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        y_hat_argmax = torch.argmax(torch.softmax(y_hat,dim=1),dim=1)
        
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
        
        
        log_dict = {
            "val_loss":loss,
            "val_acc":tm_acc,
            "val_iou":tm_iou,
            "val_f1":tm_f1,
            "val_precision":tm_prec,
            "val_recall":tm_rec,
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
        
        
#         self.log("val_tm_iou_class",tm_iou_class,prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)