import os
import imageio
from matplotlib import pyplot as plt
import cv2
import einops
import numpy
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
from video_diffusion_pytorch__.video_diffusion_pytorch import seek_all_images
import h5py
import torch.nn.functional as F
def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
    


class Get_tager_sample_h5npy(Dataset):
    def __init__(self,path):        
        with open(path, 'r') as validation_file:
            self.name = [line.strip() for line in validation_file.readlines()]

        self.t = transforms.Compose([
            # torchvision.transforms.Resize(size=(128, 128)),
            torchvision.transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        x = numpy.load(os.path.join("/media/ybxy/c89da59f-580c-440d-bab8-554bd51bb407/test_h5",self.name[idx]))
 
        x = self.t(x)
      
        image_array = einops.rearrange(x,"(c t) w h -> c t w h",t=20)
        # t c w h
        
        output_tensor = F.interpolate(image_array, size=(32, 32), mode='bilinear', align_corners=False)
        return output_tensor [:,:4,...],self.name[idx]
        # return gif[:,:4,...],gif[:,4:,...]
            # ch,time,h,w
      
    def __len__(self):
        return len(self.name)
class Get_tager_sample_h5(Dataset):
    def __init__(self,path):
        self
        self.end_path = []
        
        self.data = h5py.File("/media/ps/code/output.h5", 'r', libver='latest', swmr=True, rdcc_nbytes=0, rdcc_w0=1)
        self.name = list(self.data.keys())
        
        with open(path, 'r') as validation_file:
            self.name = [line.strip() for line in validation_file.readlines()]

        self.t = transforms.Compose([
            # torchvision.transforms.Resize(size=(128, 128)),
            torchvision.transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        x = h5py.File("/media/ps/code/output.h5", "r")[self.name[idx]][:]
        x = einops.rearrange(x," t w h  -> w h t")
     
        x = self.t(x)
      
        image_array = einops.rearrange(x,"(c t) w h -> c t w h",t=20)
        
        output_tensor = F.interpolate(image_array, size=(16, 16), mode='bilinear', align_corners=False)
        return output_tensor [:,:4,...],output_tensor [:,4:,...]
        # return gif[:,:4,...],gif[:,4:,...]

    def __len__(self):
        return len(self.name)

def save_images(images, path, **kwargs):
    imggrid=[]

    for i in range(images.shape[0]):
        imggrid.append(torch.cat([images[i,0,...],images[i,1,...],images[i,2,...],images[i,3,...]],dim=1))

    img = torch.stack(imggrid,dim=0)
    img = einops.rearrange(img,'c w h -> (c w) h')
    img = img.to('cpu').numpy()
    plt.imshow(img,vmax=10,vmin=0,cmap="jet")
    plt.savefig(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def maskedata():
    class Get_tager_sample(Dataset):
        def __init__(self):
            self.img_path = os.listdir("/media/ybxy/code/U2net/train_1k")

        def __getitem__(self, idx):
            img_name = self.img_path[idx]
            radar = numpy.load(os.path.join("/media/ybxy/code/U2net/train_1k", img_name))
            # Mytrainsform(radar)
            radar = (torch.from_numpy(radar))

            # radar = torch.squeeze(radar)
            # print(radar.shape)

            tagert = radar[0:4, :, :]

            sample = radar[4:8, :, :]
            tagert = einops.rearrange(tagert, " t c w h ->  c t w h")
            sample = einops.rearrange(sample, " t c w h ->  c t w h")
            return tagert, sample

        def __len__(self):
            return len(self.img_path)
    train = Get_tager_sample()

    dataloader = DataLoader(train, batch_size=4, shuffle=True,num_workers=8,drop_last=True)
    return dataloader





def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def random_bernoulli(p:float,shape:int)-> torch.Tensor:
    x = torch.ones((shape,))
    y = torch.bernoulli(torch.tensor(p).expand((shape,)))
    x[y==0] = 0
    return x.item()


class Get_tager_sample_256(Dataset):
    def __init__(self, path):
        self.img_path = os.listdir(path)
        self.path = path

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        radar = numpy.load(os.path.join(self.path, img_name))
        start_ = (radar[0:4, :, :, :] - 0.4202) / 0.8913
        z = [5,7,9,11]
        tagert = (radar[0:4, :, :, :] - 0.4202) / 0.8913
        sample = (radar[z, :, :, :] - 0.4202) / 0.8913

        tagert = einops.rearrange(tagert, " t c w h ->  c t w h")
        sample = einops.rearrange(sample, " t c w h ->  c t w h")

        return tagert, sample

    def __len__(self):
        return len(self.img_path)
class Get_tager_sample_number(Dataset):
    def __init__(self):
        number = numpy.load("data/mnist_test_seq.npy")
        self.number = einops.rearrange(number,"t s w h -> s t w h")
    def __getitem__(self,idx):
        img = self.number[idx,:16,...]
        return ((img/255.0)).astype(numpy.float32)
    def __len__(self):
        return 10000


def savejet(img,path):
    img = torch.clip(img , -10, 10)
    img = einops.rearrange(img, "b c f h w ->  (b w) ( c f h)")
    lable = img.to('cpu').numpy()
    numpy.save("label.npy",lable)
    plt.imshow(lable, vmax=10, vmin=0, cmap="jet")
    plt.savefig(path, dpi=600, bbox_inches=0, pad_inches=0)






class Get_tager_sample_gif(Dataset):
    def __init__(self, path):
        self.img_path = os.listdir(path)
        self.path = path

        
        
    def __getitem__(self, idx):
        img_name = self.img_path[idx]

        path = os.path.join(self.path, img_name)

        gif = gif_to_tensor1(path)

        return gif[:,:4,...],gif[:,4:,...]


    def __len__(self):
        return len(self.img_path)
    
    
class Get_tager_sample_gif_path(Dataset):
    def __init__(self, path):
        self
        self.end_path = []
        with open(path, "r") as file:
            for line in file:
                # image_path = os.path.join("/media/ps/code/all_train_gif",line.strip()+".gif")  # 
                image_path = line.strip()
                self.end_path.append(image_path)
        
        self.t = transforms.Compose([
            torchvision.transforms.Resize(size=(128, 128)),
            torchvision.transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        path = os.path.join("/media/ps/code/all_train_gif",self.end_path[idx]+".gif")
        gif = gif_to_tensor1(path)
        
        return gif[:,:12,...],gif[:,12:,...]
        
        # return gif[:,:4,...],gif[:,4:,...]

    def __len__(self):
        return len(self.end_path)
    
    
    
    
class Get_tager_sample(Dataset):
    def __init__(self, path):
        self.path = os.listdir(path)

    def __getitem__(self, idx):
        img_name = self.path[idx]
        img_f = os.path.join("/media/ps/data/all_png",img_name)
        f = []
        for i in range(16):
            x = cv2.imread(os.path.join(img_f,str(i)+".jpg"))
            f.append(x)
        f = einops.rearrange(numpy.stack(f), "t w h c -> c t w h")



        return f[:,:8,...],f[:,8:16,...]
    
def save_img(x,path):
    x = einops.rearrange(x, "b c f w h ->c (b w)  (f h) ")
    tu = transforms.ToPILImage()
    x = tu(x)
    x.save(path)