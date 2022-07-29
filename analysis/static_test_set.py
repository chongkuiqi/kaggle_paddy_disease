
# 根据best模型在test set上的预测结果，统计一下各个类别的比例
# 发现和验证集上的比例差不太多，因此验证集可以不用改

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

test:
    {
        'dead_heart': 477, 'bacterial_panicle_blight': 112, 'bacterial_leaf_blight': 156,
        'brown_spot': 322, 'hispa': 537, 'downy_mildew': 196, 'blast': 585,  
        'normal': 597, 'bacterial_leaf_streak': 121, 'tungro': 366, 
    }
'''
base_dir = "./"

submission_pathname = base_dir + "submission.csv"

with open(submission_pathname, 'r') as f:
    lines = f.readlines()

    if 'label' in lines[0]:
        lines = lines[1:]

class_dict = {}
for line in lines:
    class_name = line.strip().split(',')[1]
    if class_name not in class_dict.keys():
        class_dict[class_name] = 0
    
    class_dict[class_name] += 1

print(sorted(class_dict))
