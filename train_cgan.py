from cgan import *
import sys
from GeneratePolygons import *
mode=sys.argv[1] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if mode=='mnist':
    data_dir = 'mnist_data'
    # data_loader
    img_size = 32 
    num_labels=10
    batch_size=128
    lr = 0.0002
    train_epoch = 20
    # MNIST dataset
    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, ), std=(0.5,))])

    mnist_data = datasets.MNIST(root=data_dir,
                             train=True,
                             transform=transform,
                             download=False)

    train_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                              batch_size=batch_size,
                                              shuffle=True)
    out_dir='MNIST_cDCGAN_results/' 
    model_name='MNIST_cDCGAN'
elif mode=='polygon':
    polygon_data=PolygonDataset(1500)
    img_size=32
    num_labels=len(polygon_data.sides_range)
    batch_size=128
    lr = 0.0002
    train_epoch = 20
    train_loader = torch.utils.data.DataLoader(dataset=polygon_data,batch_size=batch_size,
                                              shuffle=True)
    out_dir='polygon_cDCGAN_results/'
    model_name='polygon_cDCGAN'
# fixed noise & label
temp_z_ = torch.randn(num_labels, num_labels*num_labels)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(num_labels, 1)
for i in range(num_labels-1):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(num_labels, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)

fixed_z_ = fixed_z_.view(-1, num_labels*num_labels, 1, 1)
fixed_y_label_ = torch.zeros(num_labels*num_labels, num_labels)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = fixed_y_label_.view(-1, num_labels, 1, 1)
fixed_z_, fixed_y_label_ = Variable(fixed_z_.to(device), volatile=True), Variable(fixed_y_label_.to(device), volatile=True)

def show_result(G,num_epoch, show = False, save = False, path = 'result.png'):

    G.eval()
    test_images = G(fixed_z_, fixed_y_label_)
    G.train()

    size_figure_grid = num_labels
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False) 

    for k in range(num_labels*num_labels):
        i = k // num_labels
        j = k % num_labels
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def train_on_dataset(train_loader,root,model,num_labels,img_size): 
    # network
    G = generator(num_labels,d=128)
    D = discriminator(num_labels,d=128)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.to(device)
    D.to(device)

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr*.1, betas=(0.5, 0.999))

    # results save folder
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + 'Fixed_results'):
        os.mkdir(root + 'Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    # label preprocess
    label_range=list(range(num_labels))
    onehot = torch.zeros(num_labels, num_labels)
    onehot = onehot.scatter_(1, torch.LongTensor(label_range).view(num_labels,1), 1).view(num_labels, num_labels, 1, 1)
    fill = torch.zeros([num_labels, num_labels, img_size, img_size])
    for i in range(num_labels):
        fill[i, i, :, :] = 1

    print('training start!')
    start_time = time.time()
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []

        # learning rate decay
        if (epoch+1) == 11:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (epoch+1) == 16:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        epoch_start_time = time.time()
        y_real_ = torch.ones(batch_size)
        y_fake_ = torch.zeros(batch_size)
        y_real_, y_fake_ = Variable(y_real_.to(device)), Variable(y_fake_.to(device))
        train_discriminator=True
        for x_, y_ in train_loader:
            # train discriminator D
            D.zero_grad()

            mini_batch = x_.size()[0]

            if mini_batch != batch_size:
                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)
                y_real_, y_fake_ = Variable(y_real_.to(device)), Variable(y_fake_.to(device))

            y_fill_ = fill[y_]
            x_, y_fill_ = Variable(x_.to(device)), Variable(y_fill_.to(device))

            D_result = D(x_, y_fill_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z_ = torch.randn((mini_batch, num_labels*num_labels)).view(-1, num_labels*num_labels, 1, 1)
            y_ = (torch.rand(mini_batch, 1) * num_labels).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_]
            y_fill_ = fill[y_]
            z_, y_label_, y_fill_ = Variable(z_.to(device)), Variable(y_label_.to(device)), Variable(y_fill_.to(device))

            G_result = G(z_, y_label_)
            D_result = D(G_result, y_fill_).squeeze()

            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            if train_discriminator:
                D_train_loss.backward()
                D_optimizer.step()

            D_loss_out=D_train_loss.item()
            D_losses.append(D_loss_out)

            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, num_labels*num_labels)).view(-1, num_labels*num_labels, 1, 1)
            y_ = (torch.rand(mini_batch, 1) * num_labels).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_]
            y_fill_ = fill[y_]
            z_, y_label_, y_fill_ = Variable(z_.to(device)), Variable(y_label_.to(device)), Variable(y_fill_.to(device))

            G_result = G(z_, y_label_)
            D_result = D(G_result, y_fill_).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            G_loss_out=G_train_loss.item()
            G_losses.append(G_loss_out)
            if D_loss_out/G_loss_out<0.1:
                train_discriminator=False
            else:
                train_discriminator=True
            

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                  torch.mean(torch.FloatTensor(G_losses))))
        fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
        show_result(G,(epoch+1), save=True, path=fixed_p)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save training results")
    torch.save(G.state_dict(), root + model + 'generator_param.pkl')
    torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
    with open(root + model + 'train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

    images = []
    for e in range(train_epoch):
        img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)



train_on_dataset(train_loader,out_dir,model_name,num_labels,img_size)

