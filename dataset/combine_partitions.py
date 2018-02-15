import os
import pickle


def combine_partitions():
    data = []
    # i = 1
    print("Reading Files from /partitions Directory")

    for i, filename in enumerate(os.listdir("./partitions")):
        if filename.endswith("pkl"):
            print("Reading and Appending partition{}".format(i + 1))
            with open("./partitions/" + filename, 'rb') as f:
                data.extend(pickle.load(f))

    with open("dataset.pkl", "wb") as f:
        pickle.dump(data, f)
        f.close()
    print("Dataset was saved in your directory.")

    print("testing")
    with open("dataset.pkl", 'rb') as f:
        data = pickle.load(f)
        f.close()
    print(len(data))


def append_to_file(output_file, path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        pickle.dump(data, output_file)
        f.close()


if __name__ == "__main__":
    combine_partitions()
