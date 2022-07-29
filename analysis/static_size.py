import os
import cv2

# 查看图像的尺寸分布

# (480, 640, 3) ./dataset//val_images/brown_spot/103050.jpg
# (480, 640, 3) ./dataset//val_images/bacterial_leaf_blight/103734.jpg
# (480, 640, 3) ./dataset//train_images/brown_spot/103343.jpg
# (480, 640, 3) ./dataset//train_images/bacterial_leaf_blight/100622.jpg
# (480, 640, 3) ./dataset/test_images/200533.jpg
# (480, 640, 3) ./dataset/test_images/200916.jpg


base_dir = "./dataset/"

dataset_dir = base_dir + "test_images/"

# class_ls = os.listdir(dataset_dir)
# print(class_ls)
# class_dict = {}
# for class_name in class_ls:
#     class_dir = dataset_dir + class_name + '/'

#     imgs_ls = sorted(os.listdir(class_dir))

#     for img_name in imgs_ls:
#         img_pathname = class_dir + img_name

#         img = cv2.imread(img_pathname)

#         shape = img.shape

#         if shape[0] != 640 or shape[1] != 480 or shape[2] != 3:
#             print(shape, img_pathname)


imgs_ls = sorted(os.listdir(dataset_dir))

for img_name in imgs_ls:
    img_pathname = dataset_dir + img_name

    img = cv2.imread(img_pathname)

    shape = img.shape

    if shape[0] != 640 or shape[1] != 480 or shape[2] != 3:
        print(shape, img_pathname)
