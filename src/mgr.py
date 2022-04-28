import os
import numpy as np

from nltk.corpus import words
from src.utils.dirs import create_dirs, check_path
from src.utils.argument_parser import read_from_file, save_to_file
from numpy.random import RandomState

_base_dir = "Experiments"
_service_list = ["recognition", "kd", "evaluate"]


class Manager(object):

    def __init__(self):

        pass

    def __call__(self, runName, newRun, serviceType, randomRun, ablationType):

        assert serviceType in _service_list

        # Create a new experiment
        if newRun:

            file = open("experiments_list.txt", "r")
            exp_list = file.read().splitlines()

            if runName is not None:
                assert runName not in exp_list
                run_name = runName
            else:
                done = False
                prng = RandomState()
                while not done:
                    lens = len(words.words())
                    i = prng.randint(lens)
                    run_name = words.words()[i]
                    if run_name in exp_list:
                        continue
                    else:
                        break

            base_dir = os.path.join(_base_dir, ablationType, run_name)
            create_dirs(base_dir)
            config_file = "cfgs/dummy.yaml"
            _, exp_dict = read_from_file(config_file)

            exp_dict["runName"] = run_name
            exp_dict["newRun"] = False

            save_to_file(os.path.join(base_dir, "args.yaml"), exp_dict)

            print("\n", "*"*10, run_name, "*"*10, "\n")
            print('Edit the args file in the directory: {}'.format(base_dir)
                  + ' and run again with \nnewRun: False \n' +
                  'runName: {}'.format(run_name))
            self.write_file(run_name)
            exit(0)

        # Read from an already existing experiment
        assert runName is not None, "Run name is null"

        # Always print experiment name in the start
        print("\n")
        print("*"*25, runName, "*"*25)
        print("\n")

        base_dir = os.path.join(_base_dir, ablationType, runName, str(randomRun))
        # Modified base_dir
        create_dirs(base_dir)

        # check if experiment path exists
        assert check_path(base_dir), "Experiment doesn't exist"
        # Read arguments
        config_file = os.path.join(base_dir,  "args.yaml")
        args, _ = read_from_file(config_file)

        # Saving what's necessary
        self.exp_name = runName
        self.random_run = randomRun
        self.service_type = serviceType
        self.service_name = args.serviceName
        self.base_dir = base_dir
        self.settingsConfig = args.settingsConfig
        self.dataConfig = args.dataConfig
        self.common = args.common

    def write_file(self, run_name):

        file1 = open("experiments_list.txt", "a")  # append mode
        dumpStr = run_name + "\n"
        file1.write(dumpStr)
        file1.close()


manager = Manager()
