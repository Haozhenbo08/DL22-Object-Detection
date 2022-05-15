## barlow.py
In barlow.py, a ResNet50 is trained with unlabeled data using Barlow Twin's method. The method getTrainedBarlowModel(totalEpochs) will save and return the trained model and the model's dictionary at the folder "/barlow-50-test01/".

## dataset.py
Add a CustomDataset class to the file provided by TA to support the use of Barlow Twin's method.

## main.py
Used to run getTrainedBarlowModel(totalEpochs).

## To submit jobs from GCP
1) Connect to greene from a Linux terminal: `ssh netid@greene.hpc.nyu.edu`
2) Connect to burst: `ssh burst`
3) Connect to GCP: `srun --partition=n1s8-v100-1 --gres=gpu:1 --account csci_ga_2572_2022sp_03 --time=03:00:00 --pty /bin/bash`
4) Make a directory named barlow-backbone: `mkdir barlow-backbone`
5) Change to the barlow-backbone directory: `cd barlow-backbone`
6) Copy the files from `/scratch/rsc468` to your barlow-backbone directory: `cp -r /scratch/rsc468/barlow-backbone/* /home/netid/barlow-backbone`
7) Change line # 41 `cd /home/rsc468/barlow-backbone` in the `demo.slurm` file to match your netid
8) Submit the job: `sbatch demo.slurm`
