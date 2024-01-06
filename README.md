# FishEye Detects™

## For DTU course: 02476 Machine Learning Operations 2024
#### Overall goal of the project
The main goal of this project is to apply the tools taught in the course on a self-chosen ML project, and use this to learn hands-on about the entire ML-Ops lifecycle. The chosen project revolves around seafood type / fish species classification/detection from images.


#### The data we are going to run on (initially, may change)
The data we will use is the large scale fish dataset from kaggle, found at https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset. It contains a total of 9000 images, with 1000 images for each of 9 different types, namely:  gilt head bream, red sea bream, sea bass, red mullet, horse mackerel, black sea sprat, striped red mullet, trout, and shrimp.

Source for data:
* O.Ulucan, D.Karakaya, and M.Turkan.(2020) A large-scale dataset for fish segmentation and classification.
In Conf. Innovations Intell. Syst. Appli. (ASYU)

#### The framework we are going to use.
The project structure will be organized around several development stages. During development, we will use CookieCutter and PyTorch Lightning to develop the model. We will utilize Git, DVC, Hydra, and Weights & Biases for experiment management. To ensure good code practices, we will employ Python environments, pytest, and pylint, integrated with GitHub Actions. The code will be deployed using Docker and FastAPI, and optimized with the PyTorch Profiler.

### The models we expect to use
Initially we are going to use a CNN based model for classifiacation, but we may later include a pre-trained YOLO or Mask R-CNN from huggingface or using the pytorch ecosystem.

![ML Canvas](<reports/figures/ML Canvas.png>)

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
