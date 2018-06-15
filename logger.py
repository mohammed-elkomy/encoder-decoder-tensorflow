import time

from prettytable import PrettyTable

from training_manager import TrainingManager


class Logger:
    def __init__(self, training_manager: 'TrainingManager'):
        self.trainingManager = training_manager
        self.trainingManager.set_logger(self)


    def time_taken(self, start, end, is_corrected=True):
        correction = self.trainingManager.time_passed_correction if is_corrected else 0

        hours, rem = divmod(end - start + correction, 3600)
        minutes, seconds = divmod(rem, 60)
        return int(hours), int(minutes), int(seconds)

    def log_hyperparameters(self):
        pt = PrettyTable(['Parameter', 'Value'])
        for key, value in self.trainingManager.get_current_configs_dict().items():
            pt.add_row([key, value])

        print("Hyper parameters")
        print(pt)

    def log_training_initialized(self):
        if self.trainingManager.resume_training:
            snap = " Snapshot: " + self.trainingManager.get_current_snapshot_name() + " "
            pad = (55 - len(snap)) // 2
            print("=" * 55, "=" * 20 + "System Restored" + "=" * 20, "=" * pad + snap + "=" * pad, "=" * 55, sep="\n")
        else:
            configs = self.trainingManager.configs
            print("{} data: {} samples,each bucket has {} samples , each bucket has {} batches"
                  .format("train",
                          configs.num_samples_train,
                          configs.num_samples_train // configs.num_buckets_train,
                          configs.num_samples_train // configs.num_buckets_train // configs.batch_size))

            print("{} data: {} samples,each bucket has {} samples , each bucket has {} batches"
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
                    self.trainingManager.summary_global_step,
                    self.trainingManager.data_generator.current_epoch,
                    "{:.4f}".format(tr_accuracy),
                    "{:.4f}".format(train_loss),
                    "{:.4f}".format(t_perplexity)])
        print(pt)

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
        print(pt)

    def log_file_saved(self, saved_file, checkpoint_start_time):
        hours, minutes, seconds = self.time_taken(self.trainingManager.execution_start, time.time())
        print("{:0>2}:{:0>2}:{:0>2} : Saved file: {}".format(hours, minutes, seconds, saved_file))

        hours, minutes, seconds = self.time_taken(checkpoint_start_time, time.time(), False)
        print("Took {:0>2}:{:0>2}:{:0>2} to make and write the file".format(hours, minutes, seconds))
        print("*" * 50)

    def log_download_drive(self, _id, title, download_start_time):
        hours, minutes, seconds = self.time_taken(self.trainingManager.execution_start, time.time())
        print("{:0>2}:{:0>2}:{:0>2} : Saved file: {}".format(hours, minutes, seconds, title))

        hours, minutes, seconds = self.time_taken(download_start_time, time.time(), False)
        print("Took {:0>2}:{:0>2}:{:0>2} to ".format(hours, minutes, seconds), end="")
        print('Download and unzipped file with ID:{}, titled:{}'.format(_id, title))
        print("*" * 100)

    def log_upload_drive(self, _id, title, upload_start_time):
        hours, minutes, seconds = self.time_taken(self.trainingManager.execution_start, time.time())
        print("{:0>2}:{:0>2}:{:0>2} : Saved file: {}".format(hours, minutes, seconds, title))

        hours, minutes, seconds = self.time_taken(upload_start_time, time.time(), False)
        print("Took {:0>2}:{:0>2}:{:0>2} to ".format(hours, minutes, seconds), end="")
        print('Upload file with ID:{}, titled:{}'.format(_id, title))
        print("*" * 100)

    @staticmethod
    def log_model_restored():
        print("=" * 55, "=" * 10 + "Model trainable parameters restored" + "=" * 10, "=" * 55, sep="\n")
