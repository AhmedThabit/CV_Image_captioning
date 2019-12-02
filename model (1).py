import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        
    def forward(self, features, captions):
        # remove end tokens from captions
        embeddings = self.embed(captions[:,:-1])
        # concatenate img features and captions
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        out, hidden = self.lstm(inputs) 
        out = self.fc(out)

        return out
        
       
    def g_sample(self, inputs):        
        cap_output = []
        batch_size = inputs.shape[0]         
        hidden = self.init_hidden(batch_size) 
    
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.fc(lstm_out)  
            outputs = outputs.squeeze(1) 
            _, max_idx = torch.max(outputs, dim=1) 
            cap_output.append(max_idx.cpu().numpy()[0].item())             
            if (max_idx == 1):
                break
            
            inputs = self.word_embeddings(max_idx) 
            inputs = inputs.unsqueeze(1)             
        return cap_output
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        tokens = []
        hidden = (torch.randn(1, 1, self.hidden_size).to(inputs.device),
                  torch.randn(1, 1, self.hidden_size).to(inputs.device))
        
        for i in range(max_len):
            out, hidden = self.lstm(inputs, hidden)
            out = self.fc(out)
            token = torch.argmax(out, dim=2)    
            token_idx = token.item()            
            tokens.append(token_idx)
            
            if token_idx == 1: # break if reaches <end> token
                break
            
            # inputs next step             
            inputs = self.embed(token)
        return tokens
            
            
        
        
        