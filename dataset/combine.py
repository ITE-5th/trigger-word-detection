import os
import pickle


def combine():
    data = []
    i = 1
    print("Reading Files from /partitions Directory")

    for filename in os.listdir("./partitions")[:4]:
        if filename.endswith("pkl"):
            with open("./partitions/" + filename, 'rb') as f:
                data.extend(pickle.load(f))
                print(i)
                i = i + 1
                f.close()

    print("Writing to dataset.pkl")
    with open("dataset.pkl", "wb") as f:
        pickle.dump(data, f)
        f.close()
    print("done")


if __name__ == "__main__":
    combine()
