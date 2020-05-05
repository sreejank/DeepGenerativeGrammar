import numpy as np
import cv2


def R(theta): #rotation matrix at angle theta
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def generate_batch(batch_size,n_sides,edge_length_range=(40,60),edge_width=2,center_mean=(64,64),center_std=5):
    #batch_size: number of images to generate
    #n_sides number of sides of the full polygon (generated images may have fewer sides)
    imgs=[]
    for _ in range(batch_size):

        image=np.zeros((128,128))

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
