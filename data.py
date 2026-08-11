from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

class GameplayDatasetCNN(Dataset):
    def __init__(self,train=True):
        super().__init__()
        #Input_text
        with open(r'CNN_Training_Data\inputs.txt','r') as f:
            self.input_data = [int(a) for a in list(f.read())]
        if train:
            self.len = int(len(self.input_data) * 0.8)
        else:
            self.len = int(len(self.input_data) * 0.2)
            self.last = int(len(self.input_data) * 0.8)

        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((216,216)),
            transforms.ToTensor()
            ])

        self.train = train


    def __getitem__(self,idx):
        if not self.train:
            idx = self.last + idx
        #Images
        img_path = f"CNN_Training_Data/IMAGES/{idx}.png"
        img = Image.open(img_path)
        bw_img = self.transform(img)
        return bw_img,self.input_data[idx]

    def __len__(self):
        return self.len