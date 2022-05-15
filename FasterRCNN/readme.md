# How to run the code
Upload the code to greene, then run sbatch demo.sh. It will train a backbone model by Barlow Twin's method for only 1 epoch on unlabeled data, store it at "/backbone/model.pth". The trained model is then used to construct a Faster RCNN model. The Faster RCNN is then trained on labeled data for only 1 epoch, stored at "/frcnn.pth", and print out the evaluation result. <br />
For the model we submit, the backbone is trained for 1000 epochs. Then, the Faster RCNN is trained for 10 epochs. Here is the link for the backbone: <br />
To replicate this result, put the model.pth in "./backbone/model.pth", and revise the main.py such that model = get_model(num_classes, pretrained_backbone=True), and num_epochs = 10. 

# main.py
If model = get_model(num_classes, pretrained_backbone=True), a pretrained backbone will be loaded from "/backbone/model.pth". <br /> 
If model = get_model(num_classes, pretrained_backbone=False), a new backbone will be trained for 1 epochs and stored in "/backbone/model.pth". <br /> 
Then, with the loaded backbone, the Faster RCNN will be trained on labeled data for 1 epochs, and store the model in "/frcnn.pth".<br /> 
The default setting of main.py is model = get_model(num_classes, pretrained_backbone=False).

# barlow.py
The BarlowTwins() method is used to define a resnet50 model for Barlow Twin's method, which is then used as the backbone. <br />
The getTrainedBarlowModel(totalEpochs) method will train a ResNet50 using Barlow Twin's method, store the model in "/backbone/model.pth". It also return the model when this method is called in main.py.


