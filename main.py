import numpy as np
import sys
from src.mgr import manager
from src.dataset_loader import init_dataset
from src.services import init_service
import torch
import random
import os
import ray

# Ray Initialization
# num_cpus = os.cpu_count()
# ray.init(num_cpus = num_cpus)

def main():

    runName, newRun, serviceType, randomRun, ablationType = sys.argv[1:]
    newRun = newRun.lower() == "true"
    if runName.strip().lower() == "none":
        runName = None
    manager(runName, newRun, serviceType, randomRun, ablationType)

    # Fix Randomness
    seed = manager.settingsConfig.train.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    dataset_loader = init_dataset(manager.dataConfig.loaderName)
    service = init_service(serviceType, manager.service_name, dataset_loader)
    service()

if __name__ == "__main__":
    main()
