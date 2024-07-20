# DiffJSCC

The repository contains the code for [Diffusion-Aided Joint Source Channel Coding for High Realism Wireless Image Transmission (arxiv)](https://arxiv.org/pdf/2404.17736). 

<img src="images/Main.png" alt="overview" width="700"/>


## Installation

    git clone https://github.com/mingyuyng/DiffJSCC.git
    cd DiffJSCC

    # create environment
    conda create -n diffjscc python=3.9
    conda activate diffjscc
    pip install -r requirements.txt

## Datasets

### OpenImage dataset

Please run the script below to prepare the OpenImage dataset. Note that `awscli` needs to be installed before running this script. Please refer to [OpenImage download](https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer)

    bash prepare_OpenImage.sh

### Other datasets

We also provide other datasets including CelebAHQ512 and Kodak. The readers could download them from [Google Drive](https://drive.google.com/drive/folders/1bGWQNs_n5NUatOmRajsdQsLWfL7n3KE2?usp=drive_link).

### Data folder structure

