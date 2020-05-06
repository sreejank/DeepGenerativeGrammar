import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
def R(theta): #rotation matrix at angle theta
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def generate_batch(batch_size,n_sides,edge_length_range=(40,60),edge_width=2,center_mean=(64,64),center_std=5,img_size=128):
    #batch_size: number of images to generate
    #n_sides number of sides of the full polygon (generated images may have fewer sides)
    imgs=[]
    for _ in range(batch_size):

        image=np.zeros((img_size,img_size))

        center=center_mean+center_std*np.random.randn(2)
        edge_length=np.random.uniform(low=edge_length_range[0],high=edge_length_range[1])
        theta0=np.random.rand()*2*np.pi
        vertices=np.array([np.dot(R(theta+theta0),edge_length*np.array([1,0])) for theta in np.linspace(0,2*np.pi,n_sides,endpoint=False)])+center

        n_sides_cur=np.random.choice(np.arange(2, n_sides+1)) #number of sides for this image
        vertices=np.vstack((vertices,vertices[0]))[:n_sides_cur+1] #+1 is bc if there are 2 sides, then we need 3 points

        for v,vnext in zip(vertices[:-1],vertices[1:]):
            image=cv2.line(image, tuple(v.astype('int')), tuple(vnext.astype('int')), (255), edge_width)
        imgs.append(image)
    return np.stack(imgs)

class PolygonDataset(Dataset):
    def __init__(self,n_imgs_per_class,sides_range=[3,4,5,6,7,8,9,10]):
        self.imgs=[]
        self.labels=[]
        for class_id,n_sides in enumerate(sides_range):
            batch=generate_batch(n_imgs_per_class,n_sides,edge_width=1,center_mean=(16,16),img_size=32,center_std=1,edge_length_range=(12,14)).astype('float32')
            batch=np.expand_dims(batch,1)
            self.imgs.append(batch)
            labels=[class_id for _ in range(n_imgs_per_class)]
            self.labels.append(labels)
        self.imgs=np.vstack(self.imgs)
        self.labels=np.hstack(self.labels)
        self.n=self.labels.shape[0]
        self.sides_range=sides_range
    def __len__(self):
        return self.n
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.imgs[idx],self.labels[idx]) 














