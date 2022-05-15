## barlow.py
In barlow.py, a ResNet50 is trained with unlabeled data using Barlow Twin's method. The method getTrainedBarlowModel(totalEpochs) will save and return the trained model and the model's dictionary at the folder "/barlow-50-test01/".

## dataset.py
Add a CustomDataset class to the file provided by TA to support the use of Barlow Twin's method.

## main.py
Used to run getTrainedBarlowModel(totalEpochs).

## To submit jobs from Greene
1) Connect to greene from a Linux terminal: `ssh netid@greene.hpc.nyu.edu`
2) Clone the repositoty `git clone git@github.com:lcqsigi/deep-learning-final.git`
3) Change directory: `cd barlow-backbone-greene`
4) Change line # 21 `cd /home/rsc468/barlow-backbone-greene` in the `demo.sh` file to match your netid
5) Submit the job: `sbatch demo.sh`

