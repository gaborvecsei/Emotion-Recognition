import random
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from utils import preprocess_image, normalize_array

datagen_all = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.06,
    height_shift_range=0.06,
    shear_range=0.06,
    zoom_range=0.06,
    horizontal_flip=True,
    fill_mode='nearest')

datagen_horizontal_flip = ImageDataGenerator(horizontal_flip=True)


def augment_brightness_on_image(image_gray):
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    image_hsv = np.array(image_hsv, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image_hsv[:, :, 2] = image_hsv[:, :, 2] * random_bright
    image_hsv[:, :, 2][image_hsv[:, :, 2] > 255] = 255
    image_hsv = np.array(image_hsv, dtype=np.uint8)
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return image_gray


def add_random_shadow(image_gray):
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    top_y = image_gray.shape[0] * np.random.uniform()
    top_x = 0
    bot_x = image_gray.shape[1]
    bot_y = image_gray.shape[0] * np.random.uniform()
    image_hls = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image_rgb.shape[0], 0:image_rgb.shape[1]][0]
    Y_m = np.mgrid[0:image_rgb.shape[0], 0:image_rgb.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    # random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return image_gray


def data_generator(batch_size, X, y, image_data_generator=None, augment_brightness=False, augment_shadows=False):
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            randomIndex = random.randint(0, len(X) - 1)
            # (48, 48)
            image = X[randomIndex]
            label = y[randomIndex]
            image = preprocess_image(image)

            cv2.imshow("ASD1", image)
            if augment_brightness:
                image = augment_brightness_on_image(image)
                cv2.imshow("ASD2", image)

            if augment_shadows:
                image = add_random_shadow(image)
                cv2.imshow("ASD3", image)
            cv2.waitKey()

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
