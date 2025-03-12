# Chaotic Fractals
[Project Page](https://chaotic-fractals.mpi-inf.mpg.de/) | [Paper](https://chaotic-fractals.mpi-inf.mpg.de/Chaotic_Fractals.pdf) 

Learning Image Fractals Using Chaotic Differentiable Point Splatting.\
[Adarsh Djeacoumar](https://scholar.google.com/citations?user=3oeUgGEAAAAJ&hl=en), [Felix Mujkanovic](https://people.mpi-inf.mpg.de/~fmujkano/), [Hans-Peter Seidel](https://people.mpi-inf.mpg.de/~hpseidel/), [Thomas Leimkühler](https://people.mpi-inf.mpg.de/~tleimkue/)

![](teaser.jpg)

## Setup
```
git clone "https://github.com/sshadr/chaotic-fractals" --recursive
```

### Environment:

| Conda | pip | Comments |
|------|------|------|
| ```conda create -n "chaos" python=3.11``` | ```python -m venv chaos``` |
| *Same as pip* |`pip install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124`| Install [Pytorch](https://pytorch.org/get-started/locally/) with CUDA. Make sure that the System CUDA matches with the installation version. |
|`conda env update -f environment.yml`| `pip install -r requirements.txt`| Install all other packages |

### Dataset
We use the implementation from [Improving Fractal Pre-training](https://github.com/catalys1/fractal-pretraining) for generating fractal datasets.

We provide the dataset used in our experiments in our project page. The default organization of the files as follows:

```
├── base
│ ├── dataset
│ │ ├── full_dataset
│ │ │ ├── ifs/ # input images in .png format
│ │ │ ├── ifs_codes/ # groundtruth fractal codes
```
If you wish to generate more random fractals (for example, to create training data for "neural regression"), use `./howtorun/generate_gt.sh`. By default, `total_points_target = int(1e8/2)` and supersampling factor, `factor=16` (256x) in `./base/data_gen/ground_truth_generator.py`. Depending on the capacity of your GPU, increase or decrease these parameters.

## Running Our Experiments

We provide the settings to run various experiments in `./base/configs`. Implementations are present in the directory `./base/trainers/` and the ways to run them are in `./base/howtorun/`.

Modify the arguments in the scripts mentioned below as needed (they should be self-explanatory or commented). To run an experiment, navigate to the folder containing the bash file and execute it using `./<filename>.sh`. The provided bash script also allows running multiple training runs in parallel, depending on GPU capacity.

| Method | **Trainers** | **How to Run** | Comments |
|------|------|----------|----------|
| Ours | `trainer.py` | `run_cf.sh`|
| Evolutionary | `trainer_0order.py` | `run_stochastic.sh`| Refer to comments in .sh |
| Cuckoo Search | `trainer_0order.py` | `run_stochastic.sh`| Refer to comments in .sh |
| Neural Regression | `trainer_neural.py` | `./base/scripts/neural_regression.py`  | training and evaluation paths have to be updated. |


| Ablations | **Trainers** | **How to Run** | Comments |
|------|------|----------|---------|
| Simulated Annealing (w/o Gradients) | `trainer_sa.py` | `run_sa.sh`|
| Moments | `trainer_moment.py` | `run_moments.sh`|
|All Other Ablations | `trainer.py` | `run_cf.sh`| Config files are present in `./base/configs`. Modify the argument in `run_cf.sh`|

### Evaluation
Our evaluation pipeline as described in the paper can be run using the following scripts in `./base/scripts/` and `./base/howtorun/`:

| Scripts | How to Run | Comments |
|------|----------|---------|
| `scale_space_eval.py` | `scale_space_eval.sh` | Evaluation pipeline to measure scale-space consistency of optimized fractals. |
| `compute_metrics.py` | `compute_metrics.sh` | Computes quantitative evaluation metrics (Tab. 3, 4). |
| `fractal_zoomer.py` | `generate_fig.sh` | Generates continuous zooms similar to the video demonstration. The trajectories that the camera has to zoom can be generated using the viewer (`./base/utils/viewer.py`). Press H while running the viewer to see keybindings and available use cases.|

### Optimized Results
We provide the results (optimized fractal codes) from our experiments in our [project webpage](https://chaotic-fractals.mpi-inf.mpg.de/). These fractals can be visualized using our OpenGL-based point rasterizer.

The viewer is located in `./base/utils/viewer.py` and can be used after modifying the path to the optimized fractal codes or any arbitrary json file containing the fractal code. Examples for how to construct such dictionaries are present in the `viewer.py` file.

## FAQ
<a id="q4"></a>
**Q: Error: `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`**

Pytorch ships its own libiomp5md.dll and hence a duplicate version exists in your conda environment. Please remove/rename the one in `<conda env path>/Library/bin/`

### Acknowledgments
Our Differentiable CUDA Rasterizer is built upon the 3D Gaussian Splatting [Rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization). 

### Citation
Please cite our paper if you refer to our results or use the method or code in your own work:

    @article{
        TBD
    }