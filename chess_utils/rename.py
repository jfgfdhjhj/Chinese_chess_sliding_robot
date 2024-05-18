import os

# 获取当前目录下所有的txt文件
txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]

# 遍历每个txt文件
for txt_file in txt_files:
    # 打开txt文件并读取内容
    with open(txt_file, 'r') as f:
        content = f.read()

    # 获取文件名的第一个字母
    first_char = txt_file[0]

    if first_char.isdigit():
        new_first_char = str(int(first_char) - 1)
    else:
        new_first_char = first_char
    print(new_first_char)
    # 将文件内容的第一个字符替换为文件名的第一个字母
    new_content = new_first_char + content[1:]
    print(new_content)
    # 写入替换后的内容到txt文件
    with open(txt_file, 'w') as f:
        f.write(new_content)
