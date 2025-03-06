# Goal
Predict mask of Brain Tumor

# Dataset
[Figshare Brain Tumor Dataset](https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset)
### Data Preprocess
- Image transform:
    ```python
    train_tfm = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from the center
    ])
    ``` 

# Model
- [x]  U-net
- [x]  Conformer + CNN
- [x]  Pretrained U-net from Pytorch (pretrained on medical data)

# Result
> Predict Mask: Upper half is perdicted mask, bottom half is ground truth mask 
### Model performance
- U-net: 
    - [Predict Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet_ground_prediction.png) | 
    - [Loss and test set accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet_loss_acc.png)
- Conformer U-net:
    - [Predict Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Conformer-Unet_ground_prediction.png) | 
    - [Loss and test set accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Conformer-Unet_loss_acc.png)
- Pretrain U-net:
    - [Predict Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Pretrain_Unet_Medical_ground_prediction.png) | 
    - [Loss and test set accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Pretrain_Unet_Medical_loss_acc.png)
### Model performance with image augmentation
- Unet, adding mask to image
    - [Predict Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet-mix_ground_prediction.png) | 
    - [Loss and test set accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet-mix_loss_acc.png)
- Unet, adding noise to image 
    - [Predict Mask](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet-noise_ground_prediction.png) | 
    - [Loss and test set accuracy](https://github.com/KJJHHH/Segmentation-Brain-Tumor/blob/main/result/Unet-noise_loss_acc.png)