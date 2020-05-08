import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from GeneratePolygons import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transforms = transforms.Compose([transforms.ToTensor()])
"""
train_dataset = datasets.MNIST(
    root='mnist_data',
    train=True,
    download=False,
    transform=transforms)

test_dataset = datasets.MNIST(
    root='mnist_data',
    train=False,
    download=False,
    transform=transforms
)
"""

polygon_data=PolygonDataset(1500) 
train_dataset=polygon_data
test_dataset=polygon_data


BATCH_SIZE = 64         # number of data points in each batch
N_EPOCHS = 20           # times to run the model on complete data
INPUT_DIM = 28 * 28     # size of each input
HIDDEN_DIM = 256        # hidden dimension
LATENT_DIM = 75         # latent vector dimension
N_CLASSES = 2          # number of classes in the data
lr = 1e-3               # learning rate 

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)

def idx2onehot(idx, n=N_CLASSES):

    assert idx.shape[1] == 1
    assert torch.max(idx).item() < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.data, 1)

    return onehot

class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.fc1 = nn.Linear(input_dim + n_classes, 576)
        self.fc2= nn.Linear(576,400)
        self.fc3= nn.Linear(400,hidden_dim)

        self.mu = nn.Linear(hidden_dim, latent_dim) 
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]

        hidden1 = F.relu(self.fc1(x))
        hidden2 = F.relu(self.fc2(hidden1))
        hidden3 = F.relu(self.fc3(hidden2))

        # hidden is of shape [batch_size, hidden_dim]

        # latent parameters
        mean = self.mu(hidden3)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(hidden3)
        # log_var is of shape [batch_size, latent_dim]

        return mean, log_var

class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        #self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        #self.hidden_to_out = nn.Linear(hidden_dim, output_dim)
        self.fc1=nn.Linear(latent_dim+n_classes,150)
        self.fc2=nn.Linear(150,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,576)
        self.fc4=nn.Linear(576,output_dim)


    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        # x is of shape [batch_size, hidden_dim]
        generated_x = F.sigmoid(self.fc4(x))
        # x is of shape [batch_size, output_dim]

        return generated_x


class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_classes)

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var

# model
model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, N_CLASSES)
model=model.to(device)

#optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

def calculate_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD

def train():
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, (x, y) in enumerate(train_iterator):
        # reshape the data into [batch_size, 784]
        x = x.view(-1, 28 * 28)
        x = x.to(device)

        # convert y into one-hot encoding
        y = idx2onehot(y.view(-1, 1))
        y = y.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        reconstructed_x, z_mu, z_var = model(x, y)

        # loss
        loss = calculate_loss(x, reconstructed_x, z_mu, z_var)

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    return train_loss

def test():
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, y) in enumerate(test_iterator):
            # reshape the data
            x = x.view(-1, 28 * 28)
            x = x.to(device)

            # convert y into one-hot encoding
            y = idx2onehot(y.view(-1, 1))
            y = y.to(device)

            # forward pass
            reconstructed_x, z_mu, z_var = model(x, y)

            # loss
            loss = calculate_loss(x, reconstructed_x, z_mu, z_var)
            test_loss += loss.item()

    return test_loss

best_test_loss = float('inf')

for e in range(N_EPOCHS):

    train_loss = train()
    test_loss = test()

    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

    if best_test_loss > test_loss:
        best_test_loss = test_loss
        patience_counter = 1
    else:
        patience_counter += 1

    if patience_counter > 3:
        break

fig,axarr=plt.subplots(N_CLASSES,5)
for i in range(N_CLASSES):
	for j in range(5):
		# create a random latent vector
		z = torch.randn(1, LATENT_DIM).to(device)

		# pick randomly 1 class, for which we want to generate the data
		#y = torch.randint(0, N_CLASSES, (1, 1)).to(dtype=torch.long)
		y=torch.tensor([[i]]).to(dtype=torch.long)
		lbl=y.item()
		print(f'Generating a {lbl}')

		y = idx2onehot(y).to(device, dtype=z.dtype)
		z = torch.cat((z, y), dim=1)

		reconstructed_img = model.decoder(z)
		img = reconstructed_img.view(28, 28).data
		print(type(img))
		img=img.cpu().numpy() 

		plt.figure()
		axarr[i][j].imshow(img, cmap='gray')
		axarr[i][j].axis('off')
fig.savefig('MNIST_CVAE_results/samples.png') 























