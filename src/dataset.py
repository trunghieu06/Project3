
from torch.utils.data import Dataset
import numpy as np

CLASSES = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer",
           "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]


class MyDataset(Dataset):
    def __init__(self, root_path="data", total_images_per_class=10000, ratio=0.8, mode="train"):
        self.root_path = root_path
        self.num_classes = len(CLASSES)

        if mode == "train":
            self.offset = 0
            self.num_images_per_class = int(total_images_per_class * ratio)

        else:
            self.offset = int(total_images_per_class * ratio)
            self.num_images_per_class = int(total_images_per_class * (1 - ratio))
        self.num_samples = self.num_images_per_class * self.num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        file_ = "{}/full_numpy_bitmap_{}.npy".format(self.root_path, CLASSES[int(item / self.num_images_per_class)])
        image = np.load(file_).astype(np.float32)[self.offset + (item % self.num_images_per_class)]
        image /= 255
        return image.reshape((1, 28, 28)), int(item / self.num_images_per_class)


if __name__ == "__main__":
    training_set = MyDataset("../data", 500, 0.8, "train")
    print(training_set.__getitem__(3))
