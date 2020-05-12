import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Differentiable grammar using matrix mult production rules and gumble softmax to choose production rules.
class DifferentiableGrammar(torch.nn.Module):
	def __init__(self,N=15,max_seq_length=20,lstm_output_size=64):
		super(DifferentiableGrammar,self).__init__()
		self.N=N
		self.max_seq_length=max_seq_length 
		self.lstm_output_size=lstm_output_size
		#Visual Encoder Module
		self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
		self.maxpool1 = torch.nn.MaxPool2d(kernel_size=4)
		self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
		self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
		self.fc1=torch.nn.Linear(20,self.N)
		#Grammar Module
		self.grammar_mat=torch.randn(self.N,self.N).to(device) 
		#LSTM
		self.lstm=torch.nn.LSTM(input_size=self.N,hidden_size=64,num_layers=1,batch_first=True)
		
		self.lstm.bias_ih_l0.data.zero_()
		self.lstm.bias_hh_l0.data.zero_()
		init.xavier_uniform_(self.lstm.weight_ih_l0)
		init.xavier_uniform_(self.lstm.weight_hh_l0) 

		#Visual Decoder Module
		self.deconv1=torch.nn.ConvTranspose2d(self.lstm_output_size,32,7)
		self.deconv2=torch.nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1)
		self.deconv3=torch.nn.ConvTranspose2d(16,1,3,stride=2,padding=1,output_padding=1)
	
	def forward(self,x):
		x=self.maxpool1(self.conv1(x)) 
		x=torch.nn.ReLU()(x)
		x=self.maxpool2(self.conv2(x))
		x=torch.nn.ReLU()(x)
		x=self.fc1(torch.reshape(x,(len(x),-1)))
		one_hot=torch.nn.functional.gumbel_softmax(x,hard=True)
		sequence=[one_hot[None,:,:]]
		
		for _ in range(self.max_seq_length-1):
			one_hot[torch.where(one_hot[self.N-1]==0)]=torch.nn.functional.gumbel_softmax(
				torch.matmul(one_hot[torch.where(one_hot[self.N-1]==0)],self.grammar_mat),hard=True)
			sequence.append(one_hot[None,:,:])
		sequence=torch.cat(sequence,0)
		#lengths=torch.FloatTensor([sequence.shape[0] for _ in range(sequence.shape[1])])
		#packed=pack_padded_sequence(sequence,lengths,batch_first=False)
		x,(ht,ct)=self.lstm(sequence)
		x=torch.nn.ReLU()(self.deconv1(x[self.max_seq_length-1].reshape(-1,self.lstm_output_size,1,1))) 
		x=torch.nn.ReLU()(self.deconv2(x))
		x=torch.nn.Sigmoid()(self.deconv3(x))
		return x 
if __name__=='__main__':
	data_dir='mnist_data'
	img_size=28
	batch_size=128
	lr=0.0002
	train_epochs=10
	transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, ), std=(0.5,))])
	mnist_data = datasets.MNIST(root=data_dir,
                             train=True,
                             transform=transform,
                             download=False)
	dataloader = torch.utils.data.DataLoader(dataset=mnist_data,
                                              batch_size=batch_size,
                                              shuffle=True)
	model=DifferentiableGrammar().to(device)
	criterion=torch.nn.MSELoss()
	optimizer=torch.optim.Adam(model.parameters(),lr=lr)
	example_images=[]
	for epoch in range(train_epochs):
		losses=[]
		for i,data in enumerate(dataloader):
			img,_=data
			img=img.view(-1,1,28,28)
			img=torch.autograd.Variable(img).to(device)
			output=model(img)
			loss=criterion(output,img)
			losses.append(loss.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print('epoch [{}/{}], avg loss:{:.4f}'.format(epoch + 1, train_epochs, np.mean(losses)))
		example_images.append(output)
	fig,axarr=plt.subplots(train_epochs,10) 

	for i in range(len(example_images)): 
		images=example_images[i].cpu().detach().numpy()
		idxs=np.random.choice(list(range(len(images))),size=10,replace=False)
		for j_idx,j in enumerate(idxs):
			generated_img=images[j,0]
			axarr[i][j_idx].axis('off')
			axarr[i][j_idx].imshow(generated_img)
	fig.savefig('MNIST_Grammmar_results/samples.png') 


