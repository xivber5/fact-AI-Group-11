# FViT 
This is the code implementation for the study *Verifying the robustness and stability of Faithful Vision
Transformers*.

# Environment Setup
1. Please set up the environment 'env.yaml' by using conda. It is possible to install this environment using the premade job file called 'install_environment.job'.

2. Some experiments require Python notebook files to be conducted, which can possibly take a long time to run. It is possible to run these notebook files as a whole using a job file. In order to successfully run these job files however, please first install the fact kernel using the following commands:

""
module purge
module load 2023
module load OpenMPI/4.1.5-NVHPC-24.5-CUDA-12.1.1
module load Anaconda3/2023.07-2
source activate fact
python -m ipykernel install --user --name=fact --display-name "Python (fact)"
""

The kernel that is installed using the above 2 command lines, can also be used to run Python notebook files without job files. 


- Use this link 'https://mpi4py.readthedocs.io/en/stable/install.html' for further information about how to download 'openmpi' and 'mpi4py'.

# Downloading datasets
The study this codebase belongs to uses several datasets that the user wil have to install seperately. The second experiment uses the a ImageNet validation set called gtsegs_ijcv.mat, which can be downloaded via the following link: https://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat. After installing, gtsegs_ijcv.mat in the folder called "datasets".

The third experiment also uses a validation set from ImageNet, it can be downloaded via the following link: 
https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.

Note that this is just a magnet link and that a bittorrent client is required to get the actual tar file. After downloading, please place ILSVRC2012_img_val.tar in the folder called `ImageNetVal/`. In this folder `file_to_label.txt` contains the target labels for the images. `ImageNetVal/dataloader.py` handles unpacking the tar file and loading the data.

# Reproducing the first experiment, qualitatively evaluating stableness and robustness of attention maps
Reproducing the first experiment is very straightforward. The relevant notebook that can be use to reproduce this experiment is fvit-demo.ipynb. If desired, this Python notebook file can also be ran using the run_demo.job file. After this job file has been completed, a new Python notebook will automatically be created called 'fvit-demo.nbconvert.ipynb'. However, note that the required runtime for this experiment is low (5-10 min) compared to other experiments(1-12h), using a job file for conducting the experiment is not a must. 

## Extensions experiment 1
The extension visualisation of the first experiment is also in the fvit-demo.ipynb and is at end of the notebook. 
### Error notes
After running the whole notebook (not in a jobfile, but via VScode), be aware that re-running an extension image cell can result in a memory overload in Snellius, so the only option is to restart and run all again...

# Reproducing the second experiment, quantitatively evaluating stableness and robustness of attention maps
The second experiment can be reproduced by using the notebook 'eval_segment.ipynb'. The notebook is computationally heavy and has to be runned via Snellius. The associated job file that needs to be runned is 'run_eval_segment.job'.
- To access the different models change the self.model argument, which is described in section 3 of the notebook.
- To access the various methods change the self.method argument, which is described in section 3 of the notebook.

## Extensions experiment 2
The extension of the second experiment can be run in extension_table_1.ipynb. The notebook is divided in 3 sections : 1. Results, 2. Load libraries and 3. Calculate metrics.
### VIA NOTEBOOK:
In order to just see all results, run cell in section 1.  
In order to run the experiment 3 steps are required:
    - run section 2
    - in section 3 change METHOD and DDS
    - run section 3 and wait, results will be automatically added.
### VIA Jobfile:
    - just change METHOD and DDS and run the jobfile 'run_eval_extension.job'.

# Reproducing the third experiment, measuring robustness under peturbations by removing pixels from input image

`perturbation_test.py` handles the perturbation test for the baseline models, this file has several command line arguments:
* `--method`: sets the baseline method. Options `["rollout", "lrp", "vta", "raw_attn", "gradcam"]`
* `--mode`: sets mode to positive or negative perturbation. Options `['pos', 'neg']`
* `--subset`: subset size of ImageNetVal (4000 in our testing)
* `--radii`: number of attack radii for the PGD (7 in our testing)
* `--masks`: levels of pixels masking (9 in our testing)
* `--batchsize`: number of images to process at once. (defaults to 16)
* `--modeltp`: batchsize for the classification model. (defaults to 32)

Note that this script uses a modified version of the baselines folder called `baselines_mod/`. The scripts here have been modified to allow for batch processing. The results of this script are saved in `perturbation_results/`. The tensors in this folder are structured as `[atk_rad, pertur_lvl]` and contain classification accuracies

To perform the perturbation test for the proposed ("ours") `perturbation_test_ours.py` is used. Due to the compute costs of DDS, the attacking and denoising is done beforehand. This is done in `denoise_imagenet.py`, which saves the resulting images as tensors in `ImageNetVal/denoised/`. Here the tensors are saved in subfolders according to the attack radius. These tensors can be loaded in using `ImageNetVal/denoised/denoised_dataloader.py`. Finally, `perturbation_test_ours.py` can be ran with the following flags:
* `--mode`: sets mode to positive or negative perturbation. Options `['pos', 'neg']`
* `--masks`: levels of pixels masking (9 in our testing)
* `--batchsize`: number of images to process at once. (defaults to 16)
* `--modeltp`: batchsize for the classification model. (defaults to 32)

Note that attack radii is fixed to 7; and subset fixed to 4000.

Plotting the results of the perturbation test is done in `perturbation_plot.ipynb`.


# References

Our code implementation is based on the following awesome material:

1. https://github.com/kaustpradalab/FViT
2. https://jacobgil.github.io/deeplearning/vision-transformer-explainability
3. https://arxiv.org/abs/2005.00928
4. https://github.com/hila-chefer/Transformer-Explainability
5. https://github.com/openai/guided-diffusion
6. https://github.com/jacobgil/vit-explain
