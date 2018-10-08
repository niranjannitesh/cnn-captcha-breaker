# CAPTCHA Conv Net

* 2 Convolutional Layers - 2 Fully Connected Layers
* Training Time - Approx 5min - 99% Accuracy

This repo contains 4753 (images) dataset

# Requirements

* Python 3
* Pip 3

```$
pip3 install -r requirements.txt
```

# Building Dataset

```$
python3 dataset.py
```

This command will generate two data files
```
x.pickle
y.pickle
```

# Train the model

```$
python3 model.py
```

# Testing Pre-Trained Model

* Go to `dist` directory

```$
cd dist
```

* `test` directory inside `dist` directory contains some real captcha

* running `test.py` with an arg with directory will read the image and rename the file to actual captcha

```$
python3 test.py test
```
