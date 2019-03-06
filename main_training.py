import pandas as pd


def get_data():
    split_pre = "IJBA_sets/split"
    t_pre = "/train_"
    # comp = "/verify_comparisons_"
    # meta = "/verify_metadata_"
    ext = ".csv"

    training_set = {}
    # comparison_set = {}
    # metadata_set = {}

    for set_n in range(1, 11):
    # set_n = 1
        training_set[set_n] = pd.read_csv(split_pre + str(set_n) + t_pre + str(set_n) + ext,
                                          header=0)
        # comparison_set[set_n] = pd.read_csv(split_pre + str(set_n) + comp + str(set_n) + ext,
        #                                     header=0)
        # metadata_set[set_n] = pd.read_csv(split_pre + str(set_n) + meta + str(set_n) + ext,
        #                                   header=0)
    return training_set


if __name__ == "__main__":
    sets = get_data()
