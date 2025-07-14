
<div align="center">
    
# Diffusion-Aided Joint Source Channel Coding for High Realism Wireless Image Transmission

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](http://arxiv.org/abs/2404.17736) 

<img src="images/Main.png" alt="overview" width="800"/>

</div>

This is the official implementation of "Diffusion-Aided Joint Source Channel Coding for High Realism Wireless Image Transmission"

## Installation

```bash
git clone https://github.com/mingyuyng/DiffJSCC.git
cd DiffJSCC

# create environment
conda create -n diffjscc python=3.9
conda activate diffjscc
pip install -r requirements.txt
```

## Datasets

### OpenImage dataset

Please run the script below to prepare the OpenImage dataset. Note that `awscli` needs to be installed before running this script. Please refer to [OpenImage download](https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer)

```bash
bash prepare_OpenImage.sh
```

### Other datasets

We also provide other datasets, including CelebAHQ512 and Kodak. The readers could download them from [Google Drive](https://drive.google.com/drive/folders/1lDAwu91UgnmHDBlHMsuqan9ZgbpqQXa9?usp=sharing).

### Data folder structure

```plaintext
/data                            # Root directory
|-- /OpenImage                   # Open Image dataset
|   |-- /018ed13fabd94731.jpg             
|   |-- /00e48838f27aa1a3.jpg         
|-- /CelebAHQ_train_512          # CelebAHQ train set
|   |-- /0.png              
|   |-- /1.png 
|-- /CelebAHQ_test_512           # CelebAHQ test set
|   |-- /27000.png              
|   |-- /27001.png
|-- /Kodak                       # Kodak dataset
|   |-- /1.png              
|   |-- /2.png 
```

### Split the train and val set

Please run the following script to split the training set and validation set. The list of images will be placed in `datalist` folder

```bash
bash create_data_list.sh
```


## Train the JSCC encoder and decoder

### OpenImage dataset
    
ResNet structure

```bash
python train.py --config ./configs/train_jscc_cnn.yaml --name "jscc_cnn_openimage" --refresh_rate 1
```

SwinJSCC

```bash
python train.py --config ./configs/train_jscc_swin.yaml --name "jscc_swin_openimage" --refresh_rate 1
```

Note that the data paths in `configs/dataset/JSCC_OpenImage_train.yaml` and `configs/dataset/JSCC_OpenImage_val.yaml` need to be modified

### CelebA dataset
    
ResNet structure

```bash
python train.py --config ./configs/train_jscc_cnn_CelebA.yaml --name "jscc_cnn_CelebA" --refresh_rate 1
```

SwinJSCC

```bash
python train.py --config ./configs/train_jscc_swin_CelebA.yaml --name "jscc_swin_CelebA" --refresh_rate 1
```

Note that the data paths in `configs/dataset/JSCC_CelebA_train.yaml` and `configs/dataset/JSCC_CelebA_val.yaml` need to be modified


## Train the conditional diffusion model

### Download the weights of Stable Diffusion

```bash
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
```

### Merge the Stable Diffusion weights with the JSCC encoder and decoder

Here is an example of merging the weights of SD and ResNet-based JSCC

```bash
python scripts/make_stage2_init_weight.py \
           --cldm_config configs/model/cldm_cnn.yaml \
           --sd_weight v2-1_512-ema-pruned.ckpt\
           --jscc_weight path/to/the/weights \
           --output ./init_weights/weights_cnn.ckpt
```
    
### Train the conditional diffusion model

OpenImage dataset

```bash
python train.py --config ./configs/train_cldm.yaml --name 'cldm_cnn_OpenImage' --refresh_rate 1
```

Note that the paths within `/configs/train_cldm.yaml` need to be modified.

## Pre-trained models

We have moved all the checkpoints to HuggingFace and modified the inference code accordingly.

| Model | Link |
| ------- | ------- |
| DiffJSCC trained on OpenImage dataset with C_channel=4 (CBR=1/384)    | [HuggingFace](https://huggingface.co/Mingyuyang/DiffJSCC-OpenImage-CBR-1-384) |
| DiffJSCC trained on OpenImage dataset with C_channel=16 (CBR=1/96)    | [HuggingFace](https://huggingface.co/Mingyuyang/DiffJSCC-OpenImage-CBR-1-96) |
| DiffJSCC trained on CelebAHQ dataset with C_channel=2 (CBR=1/768)   | [HuggingFace](https://huggingface.co/Mingyuyang/DiffJSCC-CelebA-CBR-1-768) |
| DiffJSCC trained on CelebAHQ dataset with C_channel=8 (CBR=1/192)   | [HuggingFace](https://huggingface.co/Mingyuyang/DiffJSCC-CelebA-CBR-1-192) |

## Inference

Kodak dataset

```bash
python inference_cldm.py \
    --ckpt Mingyuyang/DiffJSCC-OpenImage-CBR-1-384 \
    --input ./data/Kodak \
    --output ./results/Kodak-CBR-1-384 \
    --steps 100 \
    --device cuda \
    --repeat_times 5 \
    --SNR 1 \    
    --show_lq \
    # --use_guidance --Lambda 100 --g_t_start 1001 --g_t_stop -1 --g_repeat 1 \ # uncomment this if want to apply intermediate guidance
```
            
CelebAHQ dataset

```bash
python inference_cldm.py \
    --ckpt Mingyuyang/DiffJSCC-CelebA-CBR-1-768 \
    --input ./data/CelebAHQ_test_512 \
    --output ./results/CelebA-CBR-1-768 \
    --steps 100 \
    --device cuda \
    --repeat_times 1 \
    --SNR -5 \    
    --show_lq \
    #--use_guidance --Lambda 100 --g_t_start 1001 --g_t_stop -1 --g_repeat 1 \ # uncomment this if want to apply intermediate guidance
```            

## Visualizations

### Kodak Images (CBR=1/384, SNR=1dB)

<img src="images/vis_new.png" alt="kodak" width="700"/>

### CelebAHQ Images (CBR=1/768, SNR=-5dB)

<img src="images/visualization_CelebA.png" alt="celebA" width="700"/>

## Acknowledgement 

This project is largely based on [DiffBIR](https://github.com/XPixelGroup/DiffBIR) and [SwinJSCC](https://github.com/semcomm/SwinJSCC). Thanks for their awesome work.

## Reference

> Mingyu Yang, Bowen Liu, Boyang Wang, Hun-Seok Kim, "Diffusion-Aided Joint Source Channel Coding For High Realism Wireless Image Transmission"

    @article{yang2024diffusion,
      title={Diffusion-Aided Joint Source Channel Coding For High Realism Wireless Image Transmission},
      author={Yang, Mingyu and Liu, Bowen and Wang, Boyang and Kim, Hun-Seok},
      journal={arXiv preprint arXiv:2404.17736},
      year={2024}
    }






