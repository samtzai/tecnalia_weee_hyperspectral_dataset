import set_python_path
from sklearn.metrics import f1_score
import pytorch_lightning as pl
import torch
from torch import nn
from timm.optim import create_optimizer_v2
from torch.optim import lr_scheduler
from torchmetrics.functional import mean_squared_error,accuracy
from core.simple_models import EncoderDecoder, SpectralEncoderDecoder, UNet
from core.losses import get_loss_function


class pl_model(pl.LightningModule):
    def __init__(self,params, weights):
        super().__init__()

        self.save_hyperparameters(ignore=['loss_fn'])
        self.opt=params['train']['optimizer']
        self.lr=params['train']['learning_rate']
        self.loss_name=params['train']['loss_function']
        self.eval_metric='accuracy'
        
        self.model_name = params['model']['timm_backbone_name']
        self.activation = params['model']['activation']
        self.image_size = params['model']['tile_size']
        self.num_channels = params['model']['num_channels']
        self.intermediate_layers = params['model']['additional_decoder_layer_neurons']

        self.list_classes = params['dataset']['classes']
        self.num_classes = len(self.list_classes)

        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.model_type = params['model']['model_type']
        if self.model_type == 'encoderdecoder':
            self.model = EncoderDecoder(self.num_channels, self.num_classes, activation_name=self.activation)
        elif self.model_type == 'unet':
            self.model = UNet(self.num_channels, self.num_classes, activation_name=self.activation)
        elif self.model_type == 'spectral':
            self.model = SpectralEncoderDecoder(self.num_channels, self.num_classes, activation_name=self.activation)
        else:
            raise ValueError('Model type %s not recognized' % self.model_type)
        

        self.loss_fn= get_loss_function(self.loss_name, None)

    def get_model_card(self):
        return self.model.model_config
    def forward(self,x,activation = True):
        logits = self.model(x.float(),activation)
        return logits
    
    def configure_optimizers(self):
        optimizer=create_optimizer_v2(self.model,opt=self.opt,lr=self.lr)
        # Learning rate reduction

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.5,patience=100)



        return [optimizer],[{"scheduler": scheduler, "monitor": "val_loss"}]
    
    def training_step(self,batch,batch_idx):
        
        x = batch['x']
        y = batch['y']
        # id = batch['id']

        activation=False

        logits=self.forward(x,activation)

        loss = self.loss_fn(logits, y.float())
        preds = nn.functional.softmax (logits,dim=1)
        self.log("train_loss_step",loss,prog_bar=True,on_step=True,on_epoch=True,logger=None)

        acc=accuracy(preds,y,task="multiclass",num_classes=self.num_classes)
        # f1=f1_score(y, logits.argmax(dim=1), average='macro', zero_division=1.0)
        train_results={'loss': loss, self.eval_metric: acc}
        train_results_report={'loss': loss.detach(), self.eval_metric: acc.detach()}
        
        self.training_step_outputs.append(train_results_report)

        return train_results
    
    def on_train_epoch_end(self):
        #outputs=self.training_step_outputs
        avg_train_loss=torch.tensor([x['loss'] for x in self.training_step_outputs]).mean()
        avg_train_metric=torch.tensor([x[self.eval_metric] for x in self.training_step_outputs]).mean()

        self.log("train_loss",avg_train_loss)
        self.log("train_"+self.eval_metric,avg_train_metric)

        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):

        x = batch['x']
        y = batch['y']
        # id = batch['id']

        activation=False

   
        logits=self.forward(x,activation)

        loss = self.loss_fn(logits, y.float())
       
        acc=accuracy(logits,y,task="multiclass",num_classes=self.num_classes)
        val_results={'loss': loss, self.eval_metric: acc}
        val_results_report={'loss': loss.detach(), self.eval_metric:acc.detach()}
        
        self.validation_step_outputs.append(val_results_report)
        return val_results
    
    def on_validation_epoch_end(self):
        # outputs=self.validation_step_outputs
        avg_val_loss=torch.tensor([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_val_metric=torch.tensor([x[self.eval_metric] for x in self.validation_step_outputs]).mean()

        self.log("val_loss",avg_val_loss)
        self.log("val_"+self.eval_metric,avg_val_metric)
        self.validation_step_outputs.clear()

        return {'val_loss':avg_val_loss}