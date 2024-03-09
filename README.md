# Goal


Predict mask and class of Brain Tumor

# Dataset


[Figshare Brain Tumor Dataset](https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset)
- Transform: Mask and Image
    ```python
    train_tfm = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from the center
    ])
    ``` 
- Label: 3 Class, one-hot encoding

# Model
- [x]  U-net
- [x]  Conformer + CNN
- [x]  Pretrained U-net from Pytorch
- [ ]  Pretrained U-net + self attention

# Train
- Data Transformation During training
    - Adding noise to image after 500 epochs
    - Adding Mask to Image in first 100 epochs

# Result
> Prediction Mask: Upper half is perdicted mask, bottom half is ground truth mask 
- U-net
    - [Prediction Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/BrainTumor_Main/result/Unet_ground_prediction.png), [Loss and accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/BrainTumor_Main/result/Unet_loss_acc.png)
- Conformer U-net
- Adding Mask to Image
- Adding Noise to Image
- Adding Mask and Noise to Image# Segmentation-Brain-Tumor
