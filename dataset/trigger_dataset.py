import joblib
from torch.utils.data import Dataset


class TriggerDataset(Dataset):
    def __init__(self, file_name):
        with open(file_name, "rb") as f:
            self.samples = joblib.load(f)

    def __getitem__(self, index):
        # print("item {} \t len {} \t y {}".format(index, self.samples[index][0].shape, self.samples[index][1].shape))
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    td = TriggerDataset('dataset.pkl')

    for i, (x, y) in enumerate(td):
        print(i)
