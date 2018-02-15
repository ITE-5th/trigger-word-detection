import joblib
from torch.utils.data import Dataset


class TriggerDataset(Dataset):
    def __init__(self, file_names):
        self.samples = []
        for file_name in file_names:
            with open(file_name, "rb") as f:
                self.samples.extend(joblib.load(f))

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    td = TriggerDataset('dataset.pkl')

    for i, (x, y) in enumerate(td):
        print(i)
