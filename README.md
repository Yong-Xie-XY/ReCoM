We directly provide the input and our output for the demo data, you can find them in `/demo/` and `/demo_audio/`.

You can find the **quantitative test results** provided in our paper in the `./quantitative metrics` folder.


## Getting started

The training code and the visualization code were tested on `Ubuntu 20.04.3 LTS` 

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU

### 1. Setup environment

Create conda environment:
```bash
conda create --name ReCoM python=3.7
conda activate ReCoM
```
Please install pytorch (v1.13.1).

    pip install -r requirement.txt

If there are conflicts among Python packages, please fix them or change the download source.

Please install [**MPI-Mesh**](https://github.com/MPI-IS/mesh). 

### 2. Get data

Download [**SHOW_dataset_v1.0.zip**](https://download.is.tue.mpg.de/download.php?domain=talkshow&resume=1&sfile=SHOW_dataset_v1.0.zip) 
unzip using ``for i in $(ls *.tar.gz);do tar xvf $i;done``.

Modify ``data_root`` in ``data_utils/apply_split.py`` to the dataset path and run it to apply ``data_utils/split_more_than_2s.pkl`` to the dataset.

### 3. Training
Please note that the process of loading data for the first time can be quite slow. If you have already completed the loading process, setting ``dataset_load_mode`` to ``pickle`` in ``config/[config_name].json`` will make the loading process much faster.
We need the **pre-trained models** of `wave2wec2-base960h` and `wav2vec2-xls-r-300m-phoneme` to process the audio. So, please download them and place them in the corresponding folders. 

    # 1. Train VQ-VAEs. 
    bash train_body_vq.sh
    # 2. Train gesture generator. We trained this part on two NVIDIA RTX 3090 GPUs.
    bash train_RET.sh
    # 3. Train face generator.
    bash train_face.sh

### 5. Testing

If you want to test the model, **you** **need** the `feature_extractor.pth` file (place it in `./experiments` folder). You can download it from Link [experiments.zip](https://drive.google.com/file/d/1bC0ZTza8HOhLB46WOJ05sBywFvcotDZG/view) (the version provided by TalkSHOW). Alternatively, you can use the code we provided, set `model_name = 's2g_body_ae'`, and then obtain the corresponding file through training.

Modify the arguments in ``test_body.sh``. Then

    bash test_body.sh

If you want to test the generalization ability of the model, download the BEAT2 dataset from [EMAGE page](https://pantomatrix.github.io/EMAGE/) and place it in the corresponding location, and set the "beat2_test" attribute to true.

### 5. Visualization

If you are performing visualization on a Linux system, then I recommend using **off-screen rendering**, such as `osmesa`.

Download [**smplx model**](https://drive.google.com/file/d/1Ly_hQNLQcZ89KG0Nj4jYZwccQiimSUVn/view?usp=share_link) (Please register in the official [**SMPLX webpage**](https://smpl-x.is.tue.mpg.de) before you use it.)
and place it in ``path-to-code/visualise/smplx_model``.
The videos are saved in ``./visualise/video/EDIT``:

    bash ReCoM_visualization.sh
