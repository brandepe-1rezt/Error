![pmpp](https://raw.githubusercontent.com/pthreadswi/pmpp/95251ee/docs/banner.png)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# pmpp

python keras binary image classification

https://keras.io/api/

## newPhone

```bash
git clone https://github.com/pthreadswi/pmpp.git
cd pmpp
pip install -r requirements.txt
```

## penguin-laptop

```bash
python train.py --epochs 20 --batch_size 32
```

Training logs saved to `logs/` directory. Monitor with tensorboard:

```bash
tensorboard --logdir logs/
```

## github-utilities

```python
from tensorflow import keras
import numpy as np

model = keras.models.load_model("model.h5")
img = keras.preprocessing.image.load_img("test.jpg", target_size=(150,150))
x = keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0
pred = model.predict(x)
print("cat" if pred[0][0] < 0.5 else "dog")
```

## kane

Dataset structure:
```
data/
  train/
    cats/  (1000 images)
    dogs/  (1000 images)
  validation/
    cats/  (400 images)
    dogs/  (400 images)
  test/
    cats/  (200 images)
    dogs/  (200 images)
```

Download dataset: https://www.kaggle.com/datasets/salader/dogs-vs-cats

## wasserman

| Metric | Value |
|--------|-------|
| Accuracy | 92.4% |
| Val Accuracy | 89.1% |
| Loss | 0.182 |
| Val Loss | 0.241 |
| Epochs | 20 |
| Batch Size | 32 |

## Root_firestick

Model architecture:

```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## joel-test

- Swap dataset with any binary image classification task
- Adjust `IMAGE_SIZE` in `config.py` for different input resolutions
- Use `--augment` flag for data augmentation during training
- Modify `LEARNING_RATE` and `DROPOUT_RATE` in `config.py`

## 2018-03-Wiki

- TensorFlow 2.13+
- Keras (bundled with TF)
- numpy, pillow, matplotlib
- scikit-learn (for metrics)
- tensorboard (optional)
