# import pandas as pd
from classes import A2TestDataSet, A2VerifyDataSet


def get_data():
    split_pre = "IJBA_sets/split"
    t_pre = "/train_"
    comp = "/verify_comparisons_"
    meta = "/verify_metadata_"
    ext = ".csv"
    ext2 = "_clean.csv"

    # col_names = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE', 'MEDIA_ID', 'SIGHTING_ID']

    training_set = {}
    testing_set = {}
    # comparison_set = {}
    # metadata_set = {}

    for set_n in range(1, 11):
    # set_n = 1
        training_set[set_n] = A2TestDataSet(split_pre + str(set_n) + t_pre + str(set_n) + ext2)
        testing_set[set_n] = A2VerifyDataSet(comp_file=split_pre + str(set_n) + comp + str(set_n) + ext,
                                             meta_file=split_pre + str(set_n) + meta + str(set_n) + ext)
    #     training_set[set_n] = pd.read_csv(split_pre + str(set_n) + t_pre + str(set_n) + ext,
    #                                       header=0,
    #                                       usecols=col_names).astype(object)
    #     comparison_set[set_n] = pd.read_csv(split_pre + str(set_n) + comp + str(set_n) + ext,
    #                                         header=0)
    #     metadata_set[set_n] = pd.read_csv(split_pre + str(set_n) + meta + str(set_n) + ext,
    #                                       header=0)
    return training_set, testing_set


if __name__ == "__main__":

    # new_ds = A2TestDataSet("IJBA_sets/split1/train_1.csv")
    training, testing = get_data()

    # for s in ts:
    #     data = ts[s]
    #     l = data.__len__()
    #     print("Split {}.".format(s))
    #     # for i in range(l):
    #     #     _, _ = data.__getitem__(i)
    #     data.clean(s)

    cnt = 0

    tdata = testing[1]

    l = tdata.__len__()
    for i in range(l):
        cnt += 1
        if cnt%100 == 0:
            t1, t2, cond = tdata.__getitem__(i)
            print("List for template 1: {0} having length: {1}".format(tdata.template_1[i], len(t1)))
            print(t1)
            print("List for template 2: {0} having length: {1}".format(tdata.template_2[i], len(t2)))
            print(t2)
            print("Label =", cond, "\n")

