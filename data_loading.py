from PIL import Image
import numpy as np
import os
import imageio
import random

IMG_SIZE = 512


def read_labels(file_path):
    data_dict = {}
    f = open(file_path, "r")
    file_contents = f.read()
    file_contents = file_contents.split('\n')
    for i in range(len(file_contents)-1):
        file_contents[i] = file_contents[i].split(',')
        data_dict[file_contents[i][0]] = file_contents[i][1]
    return data_dict


def rename_files(input_dir):
    output_dir = 'labeled_'+input_dir
    os.mkdir(output_dir)
    file_dict = read_labels(os.path.join(input_dir, input_dir+'_labels.csv'))
    breeds = file_dict.values()
    breed_set = set(breeds)
    counting_dict = {}
    for i in breed_set:
        counting_dict[i] = 0

    for img in os.listdir(input_dir):
        if img[-3:] == 'csv':
            continue
        label = file_dict[str(img)]
        counting_dict[label] += 1
        path = os.path.join(input_dir, img)
        save_name = label + '-' + str(counting_dict[label]) + '.jpg'
        image_data = np.array(Image.open(path))
        imageio.imwrite(os.path.join(output_dir, save_name), image_data)


def label_img(name):
    word_label = name.split('-')[0]
    if word_label == 'tigerlily':
        return np.array([1, 0, 0, 0, 0])
    elif word_label == 'snowdrop':
        return np.array([0, 1, 0, 0, 0])
    elif word_label == 'fritillary':
        return np.array([0, 0, 1, 0, 0])
    elif word_label == 'bluebell':
        return np.array([0, 0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 0, 1])


def load_data(mode, augment=False, shuffle=False):
    dir = 'labeled_'+mode
    if not os.path.isdir(dir):

        rename_files(mode)
    data = []
    for img in os.listdir(dir):
        label = label_img(img)
        path = os.path.join(dir, img)
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        data.append([np.array(img), label])

        if augment:
            # Basic Data Augmentation - Horizontal Flipping
            flip_img = Image.open(path)
            flip_img = flip_img.convert('RGB')
            flip_img = flip_img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            flip_img = np.array(flip_img)
            flip_img = np.fliplr(flip_img)
            # flip_img = flip_img/255.0
            data.append([flip_img, label])
    if shuffle:
        random.Random(1).shuffle(data)
    images = np.array([i[0] for i in data])
    labels = np.array([i[1] for i in data])
    return images, labels


def load_train_test():
    train_x, train_y = load_data('train', augment=True, shuffle=True)
    test_x, test_y = load_data('test')
    return train_x, train_y, test_x, test_y
