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
save_dir = 'MNIST_Grammar_results/'

# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Plot losses
def plot_loss(d_losses, g_losses, g_recon,g_advers,num_epoch, save=False, save_dir='MNIST_Grammar_results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator Total')
    plt.plot(g_recon,label='Generator Reconstruction')
    plt.plot(g_advers,label='Generator Adverserial')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'MNIST_DCGAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_result(generator, fixed_loader, num_epoch, save=False, save_dir='MNIST_Grammar_results/', show=False, fig_size=(10, 2)): 
    generator.eval()

    for images,labels in fixed_loader:
        noise=images
        break

    noise = Variable(noise.cuda())
    gen_image = generator(noise)
    gen_image = denorm(gen_image)

    generator.train()
    n_rows = 10
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):

        axes[i][0].axis('off')
        axes[i][1].axis('off')
        axes[i][0].set_adjustable('box')
        axes[i][1].set_adjustable('box')
        axes[i][0].imshow(noise[i].cpu().data.view(image_size, image_size).numpy(), cmap='gray', aspect='equal')
        axes[i][1].imshow(gen_image[i].cpu().data.view(image_size, image_size).numpy(), cmap='gray', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch+1)
    fig.text(0.5, 0.04, title, ha='center') 

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'MNIST_DCGAN_epoch_{:d}'.format(num_epoch+1) + '.png'
        print("SAVING IN ",save_fn)
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

mnist_data = dsets.MNIST(root=data_dir,
                         train=True,
                         transform=transform,
                         download=False)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=batch_size,
                                          shuffle=True)

# Models
G = Generator(num_filters, G_output_dim,N=12,max_seq_length=12,lstm_output_size=100)
D = Discriminator(D_input_dim, num_filters[::-1], D_output_dim)
G.cuda()
D.cuda()

# Loss function
criterion_adverserial = torch.nn.BCELoss()
criterion_reconstruction = torch.nn.MSELoss() 

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate*0.1, betas=betas)

# Training GAN
D_avg_losses = []
G_avg_losses = []
G_avg_recons=[]
G_avg_adverss=[]

# Fixed noise for test
num_test_samples = 10

#idxs_tensor=torch.LongTensor(num_test_samples).random_(0, 12)

#fixed_noise = to_one_hot(idxs_tensor,n_dims=12)
#fixed_noise.shape

fixed_loader=torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=num_test_samples,
                                          shuffle=True)

train_D=True

for epoch in range(num_epochs):
    D_losses = []
    G_losses = []
    G_recons=[]
    G_advers=[]
    # minibatch training
    for i, (images, _) in enumerate(data_loader):
        # image data
        mini_batch = images.size()[0]
        x_ = Variable(images.cuda())
        # labels
        y_real_ = Variable(torch.ones(mini_batch).cuda())
        y_fake_ = Variable(torch.zeros(mini_batch).cuda())

        # Train discriminator with real data
        D_real_decision = D(x_).squeeze()
        # print(D_real_decision, y_real_)
        D_real_loss = criterion_adverserial(D_real_decision, y_real_)

        # Train discriminator with fake data
        #z_ = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
        #idxs_tensor=torch.LongTensor(mini_batch).random_(0, 12)
        #z_=to_one_hot(idxs_tensor,n_dims=12)
        #z_ = Variable(z_.cuda())
        gen_image = G(x_)

        D_fake_decision = D(gen_image).squeeze()
        D_fake_loss = criterion_adverserial(D_fake_decision, y_fake_)

        # Back propagation
        D_loss = D_real_loss + D_fake_loss
        D.zero_grad()
        #if train_D:
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        #z_ = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
        #idxs_tensor=torch.LongTensor(mini_batch).random_(0, 12)
        #z_=to_one_hot(idxs_tensor,n_dims=12)
        #z_ = Variable(z_.cuda())
        gen_image = G(x_)

        D_fake_decision = D(gen_image).squeeze()
        G_loss_adverserial=criterion_adverserial(D_fake_decision, y_real_)
        G_loss_reconstruction=criterion_reconstruction(gen_image,x_)
        G_loss = 0.5*G_loss_adverserial+0.5*G_loss_reconstruction

        # Back propagation
        D.zero_grad()
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())
        G_recons.append(G_loss_reconstruction.item())
        G_advers.append(G_loss_adverserial.item())
        
        if D_loss.item()/(G_loss.item()+1e-8)<0.1:
            train_D=False
        else:
            train_D=True

        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f [recon: %.4f , adverserial: %.4f]'
              % (epoch+1, num_epochs, i+1, len(data_loader), D_loss.item(), G_loss.item(),G_loss_reconstruction.item(),G_loss_adverserial.item()))

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))
    G_avg_recon = torch.mean(torch.FloatTensor(G_recons))
    G_avg_advers = torch.mean(torch.FloatTensor(G_advers))

    torch.save(G.state_dict(),'/scratch/gpfs/sreejank/MNIST_Grammar_checkpoints/'+str(epoch))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)
    G_avg_recons.append(G_avg_recon)
    G_avg_adverss.append(G_avg_advers)

    plot_loss(D_avg_losses, G_avg_losses,G_avg_recons,G_avg_adverss, epoch, save=True)

    # Show result for fixed noise
    plot_result(G, fixed_loader, epoch, save=True, fig_size=(5, 5))

# Make gif
loss_plots = []
gen_image_plots = []
for epoch in range(num_epochs):
    # plot for generating gif
    save_fn1 = save_dir + 'MNIST_DCGAN_losses_epoch_{:d}'.format(epoch + 1) + '.png'
    loss_plots.append(imageio.imread(save_fn1))

    save_fn2 = save_dir + 'MNIST_DCGAN_epoch_{:d}'.format(epoch + 1) + '.png'
    gen_image_plots.append(imageio.imread(save_fn2))
imageio.mimsave(save_dir + 'MNIST_DCGAN_losses_epochs_{:d}'.format(num_epochs) + '.gif', loss_plots, fps=5)
imageio.mimsave(save_dir + 'MNIST_DCGAN_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)

