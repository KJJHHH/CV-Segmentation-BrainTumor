import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def iamge_with_mask(image, mask, alpha=0.1):
    """
    ---
    Input
    image: batch
    mask: batch
    ---
    Output
    """
    # Cover image with mask and both use some transparency
    return (image.cpu() * alpha + mask.cpu() * (1 - alpha)).cuda()

def process(i, l, m):
    """
    ---
    Input
    image: batch
    mask: batch
    ---
    Output
    """
    i = i.to(dtype=torch.uint8).to(dtype=torch.float32)
    m = m.to(dtype=torch.float32) * 255
    l = F.one_hot(l.to(dtype=torch.int64).reshape(-1) - 1, 3)
    return i, l, m

def forward(model, i, l, m):
    
    # Model
    pred_m, out = model(i.to(dtype=torch.float32))
    
    """
    pred_m scale: (0, 1)
    """
    loss_m = torch.norm(m/255 - pred_m, p = 2)   
    loss_l = F.cross_entropy(l.to(dtype=torch.float32), out)  
    # print(loss_m.item(), loss_l.item()) 
    """
    m_ = F.normalize(m.to(dtype=torch.float32), p=2, dim=1)
    loss_m = -torch.dot(torch.flatten(pred_m), torch.flatten(m_)) / len(torch.flatten(pred_m)) 
    """    
    loss = loss_m + loss_l*10
    return loss, pred_m, out

def add_noise(images, var_low = 1, var_high = 20):
    # Batched image
    noise_images = []
    for image in images:
        var = np.random.randint(5, 500)
        noise = torch.tensor(np.random.normal(0, var, image.shape)).to(device)
        noise_images.append(image + noise)
    return torch.stack(noise_images)