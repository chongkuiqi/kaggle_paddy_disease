import os

'''
total:
    {'dead_heart': 1442, 'bacterial_panicle_blight': 337, 'bacterial_leaf_blight': 479, 
    'brown_spot': 965, 'hispa': 1594, 'downy_mildew': 620, 'blast': 1738, 
    'normal': 1764, 'bacterial_leaf_streak': 380, 'tungro': 1088}
train:
    {'dead_heart': 1009, 'bacterial_panicle_blight': 236, 'bacterial_leaf_blight': 335, 
    'brown_spot': 675, 'hispa': 1116, 'downy_mildew': 434, 'blast': 1217, 
    'normal': 1234, 'bacterial_leaf_streak': 266, 'tungro': 762}

val:
    {'dead_heart': 433, 'bacterial_panicle_blight': 101, 'bacterial_leaf_blight': 144, 
    'brown_spot': 290, 'hispa': 478, 'downy_mildew': 186, 'blast': 521, 
    'normal': 530, 'bacterial_leaf_streak': 114, 'tungro': 326}
'''
base_dir = "./"

dataset_dir = base_dir + "/val_images/"

class_ls = os.listdir(dataset_dir)
print(class_ls)
class_dict = {}
for class_name in class_ls:
    class_dir = dataset_dir + class_name + '/'

    num_class = len(os.listdir(class_dir))

    class_dict[class_name] = num_class

print(f"类别个数:{len(class_ls)}")
print(class_dict)