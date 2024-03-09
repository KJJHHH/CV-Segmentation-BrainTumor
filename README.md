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
- Label: 3 Class, class of tumor

# Model
- [x]  U-net
- [x]  Conformer + CNN
- [x]  Pretrained U-net from Pytorch: pretrained on medical data
- [ ]  Pretrained U-net + self attention

# Train
- Data Transformation During training
    - Adding noise to image after 500 epochs
    - Adding Mask to Image in first 100 epochs

# Result
> Prediction Mask: Upper half is perdicted mask, bottom half is ground truth mask 
- U-net: 
    [Prediction Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet_ground_prediction.png) | 
    [Loss and accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet_loss_acc.png)
- Conformer U-net:
    [Prediction Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Conformer-Unet_ground_prediction.png) | 
    [Loss and accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Conformer-Unet_loss_acc.png)
- Pretrain U-net:
    [Prediction Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Pretrain_Unet_Medical_ground_prediction.png) | 
    [Loss and accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Pretrain_Unet_Medical_loss_acc.png)
- Adding Mask to Image (From Unet)
    [Prediction Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet-mix_ground_prediction.png) | 
    [Loss and accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet-mix_loss_acc.png)
- Adding Noise to Image (From Unet)
    [Prediction Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet-noise_ground_prediction.png) | 
    [Loss and accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet-noise_loss_acc.png)