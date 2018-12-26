import sys
import time

from prettytable import PrettyTable

from training_manager import TrainingManager


class Logger:
    def __init__(self, training_manager: 'TrainingManager'):
        self.trainingManager = training_manager

        self.streams = [sys.stdout, training_manager.stdout]

        self.trainingManager.set_logger(self)

    def print_all(self, *value, sep=" ", end="\n"):
        for stream in self.streams:
            print(*value, file=stream, sep=sep, end=end)
            stream.flush()

    def time_taken(self, start, end, is_corrected=True):
        correction = self.trainingManager.time_passed_correction if is_corrected else 0

        hours, rem = divmod(end - start + correction, 3600)
        minutes, seconds = divmod(rem, 60)
        return int(hours), int(minutes), int(seconds)

    def log_hyperparameters(self):
        pt = PrettyTable(['Parameter', 'Value'])
        pt.add_row(["project", self.trainingManager.project_name])
        for key, value in self.trainingManager.get_current_configs_dict().items():
            pt.add_row([key, value])

        self.print_all("Hyper parameters")
        self.print_all(pt)

    def log_training_initialized(self, is_updated):
        if self.trainingManager.resume_training:
            snap = " Snapshot: " + self.trainingManager.get_current_snapshot_name() + " "
            pad = (100 - len(snap)) // 2
            self.print_all("=" * 100, "=" * 42 + "System Restored" + "=" * 43, "=" * pad + snap + "=" * (pad + 1), "=" * 100, sep="\n")

            if is_updated:
                self.print_all("=" * 40 + " Parameters Updated " + "=" * 40, "=" * 100, sep="\n")
                self.log_hyperparameters()

        else:
            configs = self.trainingManager.configs
            self.print_all("{} data: {} samples,each bucket has {} samples , each bucket has {} batches"
                           .format("train",
                                   configs.num_samples_train,
                                   configs.num_samples_train // configs.num_buckets_train,
                                   configs.num_samples_train // configs.num_buckets_train // configs.batch_size))

            self.print_all("{} data: {} samples,each bucket has {} samples , each bucket has {} batches"
                           .format("test",
                                   configs.num_samples_test,
                                   configs.num_samples_test // configs.num_buckets_test,
                                   configs.num_samples_test // configs.num_buckets_test // configs.batch_size))

            self.log_hyperparameters()

    def log_training_summary(self, tr_accuracy, train_loss, t_perplexity):
        hours, minutes, seconds = self.time_taken(self.trainingManager.execution_start, time.time())
        pt = PrettyTable(["Source",
                          "Time Elapsed",
                          "step",
                          "epoch",
                          "avg minibatch acc",
                          "avg mini-batch loss",
                          "avg mini-batch perplexity"])

        pt.add_row(["Training",
                    "{:0>2}:{:0>2}:{:0>2}".format(hours, minutes, seconds),
                    self.trainingManager.global_step,
                    self.trainingManager.data_generator.current_epoch,
                    "{:.4f}".format(tr_accuracy),
                    "{:.4f}".format(train_loss),
                    "{:.4f}".format(t_perplexity)])
        self.print_all(pt)

    def log_validation(self, ts_accuracy, valid_loss, v_perplexity):
        hours, minutes, seconds = self.time_taken(self.trainingManager.execution_start, time.time())
        pt = PrettyTable(["Source",
                          "Time Elapsed",
                          "epoch",
                          "avg test-accurac",
                          "avg loss",
                          "avg perplexity"])

        pt.add_row(["Test",
                    "{:0>2}:{:0>2}:{:0>2}".format(hours, minutes, seconds),
                    self.trainingManager.data_generator.current_epoch,
                    "{:.4f}".format(ts_accuracy),
                    "{:.4f}".format(valid_loss),
                    "{:.4f}".format(v_perplexity)])
        self.print_all(pt)

    def log_file_saved(self, saved_file, checkpoint_start_time):
        hours, minutes, seconds = self.time_taken(self.trainingManager.execution_start, time.time())
        self.print_all("{:0>2}:{:0>2}:{:0>2} : Saved file: {}".format(hours, minutes, seconds, saved_file))

        hours, minutes, seconds = self.time_taken(checkpoint_start_time, time.time(), False)
        self.print_all("Took {:0>2}:{:0>2}:{:0>2} to make and write the file".format(hours, minutes, seconds))
        self.print_all("*" * 100)

    def log_download_drive(self, _id, title, download_start_time):
        hours, minutes, seconds = self.time_taken(download_start_time, time.time(), False)
        self.print_all("Took {:0>2}:{:0>2}:{:0>2} to ".format(hours, minutes, seconds), end="")
        self.print_all('Download and unzipped file with ID:{}, titled:{}'.format(_id, title))
        self.print_all("*" * 100)

    def log_upload_drive(self, _id, title, upload_start_time):
        hours, minutes, seconds = self.time_taken(upload_start_time, time.time(), False)
        self.print_all("Took {:0>2}:{:0>2}:{:0>2} to ".format(hours, minutes, seconds), end="")
        self.print_all('Upload file with ID:{}, titled:{}'.format(_id, title))
        self.print_all("*" * 100)

    def log_model_restored(self):
        self.print_all("=" * 100, "=" * 32 + "Model trainable parameters restored" + "=" * 33, "=" * 100, sep="\n")
