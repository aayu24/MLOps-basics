import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

# Reference - https://lightning.ai/docs/pytorch/stable/data/datamodule.htm
class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        #download the dataset and load the train/val split
        cola_ds = load_dataset('glue','cola')
        self.train_ds = cola_ds['train']
        self.val_ds = cola_ds['validation']
        #TODO: load test dataset
    
    def tokenize_data(self,example):
        #TODO: checkout max length padding
        '''
        Example is an element of Dataset which contains 3 attributes - sentence, label, idx
        Passing sentence through tokenizer will return an dictionary with 3 items - ["input_ids","token_type_ids","attention_mask"]
        '''
        return self.tokenizer(example["sentence"])
    
    def setup(self, stage: str = 'None'):
        if stage == "fit" or stage is 'None':
            # Refer to huggingface documentation - https://huggingface.co/docs/datasets/use_dataset
            self.train_ds = self.train_ds.map(self.tokenizer,batched=True)
            # Set dataset to be compatible with framework which in our case is pytorch
            self.train_ds = self.train_ds.set_format(type="torch",columns=["input_ids","attention_mask","label"])

            self.val_ds = self.val_ds.map(self.tokenizer, batched=True)
            self.val_ds = self.val_ds.set_format(format="torch",columns=["input_ids","attention_mask","label"]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_ds,self.batch_size,shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds,self.batch_size,shuffle=False)
    
    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, self.batch_size,shuffle=False)