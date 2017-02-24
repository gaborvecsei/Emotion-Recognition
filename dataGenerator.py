import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from utils import preprocess_image, normalize_array

# Example data generators:

datagen_all = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.06,
    height_shift_range=0.06,
    shear_range=0.06,
    zoom_range=0.06,
    horizontal_flip=True,
    fill_mode='nearest')

datagen_horizontal_flip = ImageDataGenerator(horizontal_flip=True)


def data_generator(batch_size, X, y, image_data_generator=None):
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            randomIndex = random.randint(0, len(X) - 1)
            # (48, 48)
            image = X[randomIndex]
            label = y[randomIndex]
            image = preprocess_image(image)

            if image_data_generator is not None:
                # Extend the dimensions for data augmentation: (1, 1, 48, 48)
                image = np.reshape(image, (1, 1,) + image.shape)

                # augments data
                # yields (1, 1, 48, 48) images
                i = 0
                for augmented_image in image_data_generator.flow(image, batch_size=1):
                    processed_for_training = normalize_array(augmented_image.reshape(1, 48, 48))
                    batch_X.append(processed_for_training)
                    batch_y.append(label)
                    i += 1
                    if i == 1:
                        break
            else:
                batch_X.append(image)
                batch_y.append(label)

        yield np.array(batch_X), np.array(batch_y)
