import torch
import torch.nn.init as init
from torch.autograd import Variable



#Differentiable grammar using matrix mult production rules and gumble softmax to choose production rules.
class DifferentiableGrammar(torch.nn.Module):
    def __init__(self,N=12,max_seq_length=12,lstm_output_size=100):
        super(DifferentiableGrammar,self).__init__()
        self.N=N
        self.max_seq_length=max_seq_length 
        self.lstm_output_size=lstm_output_size

        #Grammar Module
        self.grammar_mat=torch.randn(self.N,self.N).cuda()
        #LSTM
        self.lstm=torch.nn.LSTM(input_size=self.N,hidden_size=self.lstm_output_size,num_layers=1,batch_first=True)
        
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()
        init.xavier_uniform_(self.lstm.weight_ih_l0)
        init.xavier_uniform_(self.lstm.weight_hh_l0) 

    
    def forward(self,one_hot):
        sequence=[one_hot[None,:,:].clone()] 
        
        for _ in range(self.max_seq_length-1): 
            idxs_nonterminal=torch.where(one_hot[:,self.N-1]==0)
            if idxs_nonterminal[0].shape[0]>0:
                one_hot[idxs_nonterminal]=torch.nn.functional.gumbel_softmax(
                    torch.matmul(one_hot[idxs_nonterminal],self.grammar_mat),hard=True) 
            sequence.append(one_hot[None,:,:].clone())
        sequence=torch.cat(sequence,0)  
        self.sequence=sequence.clone()

        
        #lengths=torch.FloatTensor([sequence.shape[0] for _ in range(sequence.shape[1])])
        #packed=pack_padded_sequence(sequence,lengths,batch_first=False)
        x,(ht,ct)=self.lstm(sequence)
        return x[self.max_seq_length-1]

# Generator model
class Generator(torch.nn.Module):
    def __init__(self, num_filters, output_dim,N=12,max_seq_length=12,lstm_output_size=100):
        super(Generator, self).__init__()
        self.grammar=DifferentiableGrammar(N=N,max_seq_length=max_seq_length,lstm_output_size=lstm_output_size)
        input_dim=lstm_output_size
        self.input_dim=input_dim
        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                deconv = torch.nn.ConvTranspose2d(input_dim, num_filters[i], kernel_size=4, stride=1, padding=0)
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

            deconv_name = 'deconv' + str(i + 1)
            self.hidden_layer.add_module(deconv_name, deconv)

            # Initializer
            torch.nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(deconv.bias, 0.0)

            # Batch normalization
            bn_name = 'bn' + str(i + 1)
            self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, one_hot):
        x=self.grammar(one_hot)
        h = self.hidden_layer(x.view(-1,self.input_dim,1,1))
        out = self.output_layer(h)
        return out


# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(Discriminator, self).__init__()

        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                conv = torch.nn.Conv2d(input_dim, num_filters[i], kernel_size=4, stride=2, padding=1)
            else:
                conv = torch.nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

            conv_name = 'conv' + str(i + 1)
            self.hidden_layer.add_module(conv_name, conv)

            # Initializer
            torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(conv.bias, 0.0)

            # Batch normalization
            if i != 0:
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Convolutional layer
        out = torch.nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out
