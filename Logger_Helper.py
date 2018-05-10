import time

from prettytable import PrettyTable


def print_hyper(param_dict):
    t = PrettyTable(['Parameter', 'Value'])
    for key, value in param_dict.items():
        t.add_row([key, value])

    print("Hyper parameters")
    print(t)


def time_taken(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    # print("this took {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    return int(hours), int(minutes), seconds


def training_summary_log(step, current_epoch, tr_accuracy, loss_, t_perplexity, start_time):
    hours, minutes, seconds = time_taken(start_time, time.time())
    print("{:0>2}:{:0>2}:{:05.2f} : (Training) step :{}, epoch :{},"
          "avg minibatch-accuracy :{:.4f} ,"
          "last mini-batch loss:{:.4f} ,last mini-batch perplexity:{:.4f}".format(hours, minutes, seconds, step, current_epoch, tr_accuracy, loss_, t_perplexity))


def validation_log(step, current_epoch, ts_accuracy, valid_loss, v_perplexity, start_time):
    hours, minutes, seconds = time_taken(start_time, time.time())

    print("{:0>2}:{:0>2}:{:05.2f} : (Test) step :, {} epoch :,{},avg test-accuracy :{:.4f} ,avg loss:{:.4f} ,avg perplexity:{:.4f}"
          .format(hours, minutes, seconds, step, current_epoch, ts_accuracy, valid_loss, v_perplexity))


def file_saved_log(saved_file, start_time):
    hours, minutes, seconds = time_taken(start_time, time.time())
    print("{:0>2}:{:0>2}:{:05.2f} :Saved file: {}".format(hours, minutes, seconds, saved_file))
