import os
import re


def has_log_file(log_root):
    file_names = os.listdir(log_root)
    for file_name in file_names:
        if file_name.startswith('log'):
            return True
    return False


def find_max_log(log_root):
    files = os.listdir(log_root)
    pattern = r'log(\d+)\.pth'
    max_num = 0
    for file in files:
        match = re.match(pattern, file)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return os.path.join(log_root, f"log{max_num}.pth")


def classes_txt(root, out_path, num_class=None):
    """
    write image paths (containing class name) into a txt file.
    :param root: data set path
    :param out_path: txt file path
    :param num_class: how many classes needed
    :return: None
    """
    dirs = os.listdir(root)  # 列出根目录下所有类别所在文件夹名
    if not num_class:		# 不指定类别数量就读取所有
        num_class = len(dirs)

    if not os.path.exists(out_path):  # 输出文件路径不存在就新建
        f = open(out_path, 'w')
        f.close()
    # 如果文件中本来就有一部分内容，只需要补充剩余部分
    # 如果文件中数据的类别数比需要的多就跳过
    with open(out_path, 'r+') as f:
        try:
            end = int(f.readlines()[-1].split('\\')[-2]) + 1
        except:
            end = 0
        if end < num_class - 1:
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir in dirs:
                files = os.listdir(os.path.join(root, dir))
                for file in files:
                    f.write(os.path.join(root, dir, file) + '\n')
