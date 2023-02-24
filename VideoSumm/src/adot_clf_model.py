import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

#### Load the save model ####
class BaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        x = self.dropout(cls_token)
        output = self.linear(x)

        return output

#### predict ####
class PredictModel():
    def __init__(self,device='cpu'):
        self.device = torch.device(device)
        PATH = 'weights/classification_model.pt'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                          do_lower_case=False)
        self.model = BaseModel()
        self.model.load_state_dict(torch.load(PATH, map_location=device))
        self.model.to(device)
        

    def predict(self,data):
        text = ' '.join(data)
        
        encoded_dict = self.tokenizer.encode_plus(
            text=text, # Sequence to encode
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            max_length=512, 
            padding='max_length', # Pad and truncate
            truncation=True, #Truncate the seq
            return_attention_mask=True, # Construct attn. masks
            return_tensors='pt' # Return pytorch tensors
        )
        
        self.model.eval()
        
        input_ids = encoded_dict['input_ids'].long().to(self.device)
        input_mask = encoded_dict['attention_mask'].long().to(self.device)
        
        output = self.model(input_ids, input_mask)
        output = nn.Softmax(dim=1)(output)
        
        output = output.detach().cpu().numpy()
        
        return output[0]

    def __call__(self,data: list) -> str:
        result = self.predict(data)
        return result