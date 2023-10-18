import re
import os.path
from torch.utils.data import Dataset
import PIL
import PIL.Image as Image
from torchvision.transforms import ToTensor


class Blender(Dataset):
    def __init__(self, args, mode='train', phase_label=False):
        super(Blender, self).__init__()
        self.args = args

        self.phase_label = phase_label

        if mode == 'train':
            image_dir_list = [os.path.join(d, 'images') for d in args.data.blender_dir_list_train]
        elif mode == 'test' or mode == 'val':
            image_dir_list = [os.path.join(d, 'images') for d in args.data.blender_dir_list_test]
        else:
            raise NotImplemented

        self.image_list = []

        for dir in image_dir_list:
            image_list_i = [os.path.join(dir, fn) for fn in os.listdir(dir) if fn.endswith('png')]
            image_list_i.sort(key=lambda s: int(re.split('_|-|/|\.', s)[-2]))
            self.image_list.extend(image_list_i)

        if mode == 'val':
            self.image_list = self.image_list[:6000]
        elif mode == 'test':
            self.image_list = self.image_list[0:12000]

    def __getitem__(self, index):

        img = Image.open(self.image_list[index])
        img = img.resize((self.args.data.img_h, self.args.data.img_w), PIL.Image.BILINEAR)
        out = ToTensor()(img)[:3]

        if self.phase_label:
            return out, 0
        else:
            return out

    def __len__(self):
        return len(self.image_list)