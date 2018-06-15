import gzip
import math
import os
import pickle
import re
import shutil
import sys
import time

import tensorflow as tf
from tensorflow import Session

from EDZipFile import ZipFile
from data_generator import DataGenerator
from drive_remote_snapshots import DriveRemoteSnapshots
from model_configs import ModelConfigs


# from model_graph import ModelGraph


def encode_string(file_version_index):
    if file_version_index == 0:
        return "AAA"

    file_version_string = ""
    while file_version_index / 26 > 0:
        file_version_string = chr(ord('A') + file_version_index % 26) + file_version_string
        file_version_index //= 26

    return "A" * (3 - len(file_version_string)) + file_version_string  # 3 chars code


class BaseTrainingManager:
    """any configs not related to deep learning is placed here(management parameter)"""
    data_generator: DataGenerator

    drive_snapshot_manager: DriveRemoteSnapshots
    session: Session

    def __init__(self,
                 configs: ModelConfigs,
                 project_name: str,
                 summary_every_mins: int,
                 checkpoint_every_mins: int,
                 example_every_mins: int,
                 test_every_mins: int,
                 resume_training: bool):
        # =============================training system configs=============================
        self.project_name = project_name
        self.summary_every_mins = summary_every_mins
        self.checkpoint_every_mins = checkpoint_every_mins
        self.example_every_mins = example_every_mins
        self.test_every_mins = test_every_mins
        self.resume_training = resume_training
        # =============================non configurable=============================
        self.configs = configs

        # output files
        self.log_file = self.project_name + '.log'
        self.checkpoint_dir_name = 'checkpoints/'
        self.event_files_dir_name = "event-logs/"
        self.params_file = self.project_name + ".params"
        self.checkpoint_file_name = self.checkpoint_dir_name + self.project_name + "_"

        self.minute = 60

        self.resume_training = os.path.isfile(self.params_file) and self.resume_training
        self.is_local_env = "preferences" in os.getcwd()

        # purge if not resuming
        if not self.resume_training:
            self.purge()

        if not self.is_local_env:
            sys.stdout = open(self.log_file, 'a')

        if self.is_local_env:  # if running on my machine
            self.local_machine_protection()  # my local machine can't has only 8 gb memory and will crash :D
        # configs now suit my machine

        if not os.path.exists(self.checkpoint_dir_name):
            os.mkdir(self.checkpoint_dir_name)

        self.timestamp = str(math.trunc(time.time()))

        if not self.is_local_env:
            # folder at each run named 'log/<timestamp>/'.
            self.training_writer = tf.summary.FileWriter(self.event_files_dir_name + self.timestamp + "-training")
            self.validation_writer = tf.summary.FileWriter(self.event_files_dir_name + self.timestamp + "-validation")

        # timers
        self.execution_start = time.time()
        self.checkpoint_last = time.time()
        self.summary_last = time.time()
        self.example_last = time.time()
        self.test_last = time.time()

        # globals
        self.time_passed_correction = 0
        self.summary_global_step = 0
        self.zip_file_version = 0
        self.step = 0

        sys.stdout.flush()

    def purge(self):
        if os.path.isdir(self.checkpoint_dir_name):
            shutil.rmtree(self.checkpoint_dir_name)

        if os.path.isdir(self.event_files_dir_name):
            shutil.rmtree(self.event_files_dir_name)

        for f in os.listdir('.'):
            if re.search('.*\.zip|.*\.params|.*\.log', f):
                os.remove(f)

    def restore_system_from_snapshot(self):
        # restore parameters
        if self.resume_training and not self.is_local_env:
            # load params
            with gzip.open(self.params_file, 'rb') as f:
                configs_dict = pickle.load(f)

            for key, _ in self.configs.__dict__.items():
                if key != 'named_constants':
                    self.configs.__dict__[key] = configs_dict[key]  # from file to my data structure

            self.time_passed_correction = configs_dict["time_passed_correction"]
            self.summary_global_step = configs_dict["summary_global_step"]
            self.zip_file_version = configs_dict["zip_file_version"]

    def get_current_configs_dict(self):
        ret_dict = {
            "time_passed_correction": round(time.time() - self.execution_start + (self.time_passed_correction if self.resume_training else 0)),
            "summary_global_step": self.summary_global_step,
            "zip_file_version": self.zip_file_version
        }

        for key, _ in self.configs.__dict__.items():
            if key != 'named_constants':
                ret_dict[key] = self.configs.__dict__[key]  # from my training configs to dictionary for snapshot

        return ret_dict

    def save_params(self):
        # write params to disk
        with gzip.open(self.params_file, 'w') as fs:
            pickle.dump(self.get_current_configs_dict(), fs)

    def save_system_snapshot(self):
        # save system snapshot and compress
        file_version_string = encode_string(self.zip_file_version - 1)

        file_name = '{}_{}.zip'.format(self.project_name, file_version_string)
        snapshot_zip = ZipFile(file_name)
        snapshot_zip.add_directory(self.event_files_dir_name)
        snapshot_zip.add_directory(self.checkpoint_dir_name)

        if os.path.isfile(self.log_file):
            snapshot_zip.add_file(self.log_file)

        self.save_params()
        snapshot_zip.add_file(self.params_file)

        snapshot_zip.print_info()

        if "preferences" not in os.getcwd():
            self.drive_snapshot_manager.upload_file(file_name)

    def get_current_snapshot_name(self):
        return encode_string(self.zip_file_version - 1)

    # for datasets larger then my memory
    def local_machine_protection(self):
        raise NotImplementedError

    # will be averaged every summary
    def initialize_accumulated_variables(self):
        raise NotImplementedError

    def sustaining_callback(self, _next, graph, configs, sess):
        raise NotImplementedError

    def summary_callback(self, _next, graph, configs, sess):
        raise NotImplementedError

    def test_callback(self, graph, configs, sess):
        raise NotImplementedError

    def end_epoch_callback(self, graph, configs, sess):
        raise NotImplementedError

    def valid_example_callback(self, graph, configs, sess):
        raise NotImplementedError

    def launch_training_loop(self):
        if not hasattr(self, 'logger'):
            raise AttributeError("Training manager doesn't have logger")

        if not hasattr(self, 'graph'):
            raise AttributeError("Training manager doesn't have model/graph")

        self.initialize_accumulated_variables()
        sys.stdout.flush()
        for _next in self.data_generator.next_batch(self.configs.batch_size):
            current_time = time.time()
            is_end_of_epoch = _next[0]

            # print some examples
            if (current_time - self.example_last) > self.example_every_mins * self.minute:
                self.example_last = time.time()
                self.valid_example_callback(self.graph, self.configs, self.session)

            # sustaining step or summary step ?
            if (current_time - self.summary_last) > self.summary_every_mins * self.minute and self.step > 0:
                self.summary_last = time.time()
                self.summary_callback(_next[1:], self.graph, self.configs, self.session)

                # flush everything every summary step
                if not self.is_local_env:
                    self.training_writer.flush()
                    # noinspection PyUnboundLocalVariable
                    self.validation_writer.flush()

                # reset accumulators
                if self.step > 1000:
                    self.step = 0
                    self.initialize_accumulated_variables()
            else:
                self.sustaining_callback(_next[1:], self.graph, self.configs, self.session)

            # save a checkpoint
            if (current_time - self.checkpoint_last) > self.checkpoint_every_mins * self.minute:
                self.checkpoint_last = time.time()

                if not self.is_local_env:
                    for file in os.listdir(self.checkpoint_dir_name):
                        os.remove(os.path.join(self.checkpoint_dir_name, file))

                    # noinspection PyUnboundLocalVariable
                    saved_file = self.graph.saver.save(self.session, self.checkpoint_file_name + self.timestamp, global_step=self.summary_global_step)
                    self.zip_file_version += 1
                    self.save_system_snapshot()
                    self.logger.log_file_saved(saved_file, self.checkpoint_last)
                sys.stdout.flush()

            # apply test dataset
            if (current_time - self.test_last) > self.test_every_mins * self.minute and "preferences" not in os.getcwd():
                self.test_last = time.time()
                self.test_callback(self.graph, self.configs, self.session)

            # end of epoch
            if is_end_of_epoch:
                self.end_epoch_callback(self.graph, self.configs, self.session)
                print("asdopaspjodsawjkpd[;dsad", self.summary_global_step, "asdopaspjodsawjkpd[;dsad")

            sys.stdout.flush()
            self.step += 1
            self.summary_global_step += 1

    def set_logger(self, _logger):
        self.logger = _logger
        if "preferences" not in os.getcwd():
            self.drive_snapshot_manager = DriveRemoteSnapshots(self.project_name, _logger)
            status = self.drive_snapshot_manager.get_latest_snapshot()  # could restore
            self.resume_training = status

        if self.resume_training:  # founc data on colab and and they are compressed
            self.restore_system_from_snapshot()

        self.logger.log_training_initialized()

    def set_graph(self, model_graph):
        self.session = tf.Session(graph=model_graph.tf_graph)
        # noinspection PyAttributeOutsideInit
        self.graph = model_graph

        if self.resume_training and not self.is_local_env and len(os.listdir(self.checkpoint_dir_name)) > 0:
            checkpoint_to_restore = os.path.join(self.checkpoint_dir_name, max(os.listdir(self.checkpoint_dir_name)).split(".")[0])
            # noinspection PyUnboundLocalVariable
            model_graph.saver.restore(self.session, checkpoint_to_restore)
            self.logger.log_model_restored()
        else:
            self.session.run(self.graph.init)

    def set_data_generator(self, data_generator: DataGenerator):
        self.data_generator = data_generator

    def __del__(self):
        try:
            self.training_writer.close()
            self.validation_writer.close()
        except:
            pass
