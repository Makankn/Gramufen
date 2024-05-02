from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomDatasetTrain(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        img = Image.open(str(data.path)).convert('RGB')
        data.image = transform(img)

        data.image= data.image.reshape(int(data.image.shape[0]/3), 3, 224, 224)
        return data


class CustomDatasetTest(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        img = Image.open(str(data.path)).convert('RGB')
        data.image = transform(img)
        data.image= data.image.reshape(int(data.image.shape[0]/3), 3, 224, 224)
        return data


# resize_size = 256
# crop_size = 224
# normalize_mean = [0.485, 0.456, 0.406]
# normalize_std = [0.229, 0.224, 0.225]

# transform = transforms.Compose([
#     transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
#     transforms.CenterCrop(crop_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=normalize_mean, std=normalize_std)
# ])



train_dataset = CustomDatasetTrain(train_graphs)
val_dataset = CustomDatasetTest(val_graphs)
test_dataset = CustomDatasetTest(test_graphs)


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)