#!/usr/bin/python3.5

import time
import argparse as ap

import torch
from torch import cuda

import model as mod
from classes import Helper


def main(batch_size, num_epochs, lr, file_write, flag_dummy):
    """
    Main function for training.
    :param batch_size : Batch size to use.
    :param num_epochs : Number of epochs for each split.
    :param lr         : Learning rate to be set at the start of each split.
    :param file_write : Write output to stdout(default) or a file.
    :param flag_dummy : Create a dummy file for evaluation.
    :return:
    """

    # ''' ---------------------Parameters---------------------'''
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")
    optim_name = 'Adam'
    batch_printer = 50
    op_dir = "pickles/"
    t_stmp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    # helper = Helper()
    # num_epochs = 3
    # batch_size =

    helper = Helper("log/log_" + t_stmp + ".txt")
    helper.write_file(file_write)

    helper.log(msg="Starting data loading.")

    # training_set, training_loader, testing_set, testing_loader = helper.get_data(mode="train",
    training_set, training_loader = helper.get_data(mode="train",
                                                    training_batch_size=batch_size
                                                    )
    # model, criterion, optimizer = mod.get_model(device, 'SGD')

    helper.log(msg="Finished data loading. Starting main training.")

    # Following code only for creating a dummy file for testing pipeline.
    # flag_dummy = False
    # noinspection PyBroadException
    try:
        f = open(op_dir + "dummy.pt", 'r')
        f.close()
    except:
        flag_dummy = True

    # model.train(True)

    for set_n in range(1, 11):

        init_lr = lr

        model, criterion, optimizer = mod.get_model(device, optim_name, lamb=0, learning_rate=init_lr)
        model.train(True)
        # print(type(optimizer))

        if flag_dummy:
            helper.log(msg="\nCreating dummy file.\n")
            dummy_file = {"model": model.state_dict(),
                          "criterion": criterion.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "optim_name": optim_name
                          }
            torch.save(dummy_file, op_dir + "dummy.pt")
            flag_dummy = False

        helper.log(msg="\nStart of split {0}\n".format(set_n))

        # https://stackoverflow.com/questions/32558805/ceil-and-floor-equivalent-in-python-3-without-math-module
        total_len = -(-training_set[set_n].__len__() // batch_size)
        # total_len = training_set[set_n].__len__()
        running_loss = 0.0
        past_loss = 6.0 * batch_printer

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(training_loader[set_n]):

                # Change variable type to match GPU requirements
                inp = images.to(device)
                lab = labels.to(device)

                # a = images.size()

                # Reset gradients before processing
                optimizer.zero_grad()

                # Get model output
                out = model(inp)

                # Calculate loss
                loss = criterion(out, lab)

                # Update weights
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # logger.log(msg="\rLoss = {0}        ".format(l), end="")
                if (i + 1) % batch_printer == 0:
                    # logger.log(msg="Epoch " + str(epoch) + " Step " + str(i + 1) + "/" + str(total_len), end="\t")
                    # logger.log(msg="\rEpoch: {0}, step: {1}/{2}".format(epoch+1, i+1, total_len), end="\t")
                    helper.log(msg="Split: {3}, Epoch: {0}, step: {1}/{2}  ".format(epoch + 1, i + 1, total_len, set_n),
                               end="\t")
                    # logger.log(msg="Running Loss data: ", loss.data)
                    helper.log(msg="Running Loss (avg): {0:.08f}, Past: {1:.08f}".format((running_loss / batch_printer),
                                                                                         (past_loss / batch_printer)),
                               end="\t")

                    if running_loss < past_loss:
                        past_loss = running_loss
                        init_lr *= 0.9
                        for params in optimizer.param_groups:
                            params['lr'] = init_lr
                    # elif (i + 1) % 250 == 0:
                    #     init_lr *= 0.5
                    #     for params in optimizer.param_groups:
                    #         params['lr'] = init_lr

                    helper.log(msg="LR: {0}".format(init_lr))

                    running_loss = 0.0
                    # optimizer.zero_grad()

        filename = op_dir + "A2_T{1}_S{0}.pt".format(set_n, t_stmp)

        # Idea for named saved file was picked up from here:
        # https://github.com/quiltdata/pytorch-examples/blob/master/imagenet/main.py
        save_file = {"model": model.state_dict(),
                     "criterion": criterion.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "optim_name": optim_name
                     }

        torch.save(save_file, filename)
        helper.log(msg="\nFile {0} saved for split {1}".format(filename, set_n))

    helper.close()


if __name__ == "__main__":

    parser = ap.ArgumentParser()

    parser.add_argument("-b",
                        "--batch-size",
                        type=int,
                        default=4,
                        help="Batch Size (default 4).")

    parser.add_argument("-e",
                        "--num-epochs",
                        type=int,
                        default=3,
                        help="Number of epochs (default 3).")

    parser.add_argument("--lr",
                        "--learning-rate",
                        type=float,
                        default=0.01,
                        help="Initial learning rate (default 0.01)")

    parser.add_argument("--file-write",
                        action="store_true",
                        default=False,
                        help="Write to file instead of stdout.")

    parser.add_argument("-d",
                        "--create-dummy",
                        action="store_true",
                        default=False,
                        help="Write a dummy model file for evaluation.")

    args = parser.parse_args()

    main(args.batch_size, args.num_epochs, args.lr, args.file_write, args.create_dummy)
