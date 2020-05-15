# MNIST image generation using DCGAN
from differentiable_grammar_gan import *
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import os
import imageio 

# Parameters
image_size = 64
G_input_dim = 100
G_output_dim = 1
D_input_dim = 1
D_output_dim = 1
num_filters = [1024, 512, 256, 128]

learning_rate = 0.0002
betas = (0.5, 0.999)
batch_size = 128 
num_epochs = 100
data_dir = 'mnist_data'
save_dir = 'CLF_Grammar_results/'

# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Plot losses
def plot_loss(losses,accuracies,num_epoch, save=False, save_dir='CLF_Grammar_results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(losses, label='loss')
    plt.plot(accuracies,label='accuracy')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


# MNIST dataset
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, ), std=(0.5,))])

mnist_data_train = dsets.MNIST(root=data_dir,
                         train=True,
                         transform=transform,
                         download=False)
mnist_data_test = dsets.MNIST(root=data_dir,
                         train=False,
                         transform=transform,
                         download=False)

train_loader = torch.utils.data.DataLoader(dataset=mnist_data_train,
                                          batch_size=batch_size,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_data_test,
                                          batch_size=batch_size,
                                          shuffle=True) 

# Models
model = Classifier(num_filters, N=10,max_seq_length=3,lstm_output_size=100)
model.cuda()

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)

# Training GAN
avg_losses = []


#idxs_tensor=torch.LongTensor(num_test_samples).random_(0, 12)

#fixed_noise = to_one_hot(idxs_tensor,n_dims=12)
#fixed_noise.shape

def train(epoch):
    model.train()
    losses = []
    # minibatch training
    for i, (images, labels) in enumerate(train_loader):
        # image data
        mini_batch = images.size()[0]
        x_ = Variable(images.cuda())
        y_real=Variable(labels.cuda())

        y_hat=model(x_)
        
        loss=criterion(y_hat,y_real)

        # Back propagation
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # loss values
        losses.append(loss.item()) 
        print("Epoch [%d/%d] step [%d/%d] loss: %.4f "%(epoch,num_epochs,i+1,len(train_loader),loss.item()))

    avg_loss = torch.mean(torch.FloatTensor(losses))

    torch.save(model.state_dict(),'/scratch/gpfs/sreejank/MNIST_CLF_checkpoints/'+str(epoch))

    # avg loss values for plot
    avg_losses.append(avg_loss)

    

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            x_ = Variable(images.cuda())
            y_real=Variable(labels.cuda())
            y_hat_probs = model(x_)
            y_hat=y_hat_probs.data.max(1, keepdim=True)[1]
            correct += y_hat.eq(y_real.data.view_as(y_hat)).sum()
    print("Test Accuracy :",correct/float(len(test_loader.dataset)))



for epoch in range(num_epochs):
    train(epoch)
    test()


