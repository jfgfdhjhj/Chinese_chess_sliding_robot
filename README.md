# 中国象棋机器人，基于写字机器人的python实现

采用皮卡鱼引擎，电动滑轨xyz结构驱动的中国象棋机器人

## 大致流程
1. 棋盘寻找：调用opencv的寻找角点函数，将棋盘角点坐标保存；
2. 局面初始化：用霍夫圆检测检测棋子，将其裁剪，送入模型检测；
3. 棋子识别：采用类似minist的数据集分类模型，将原始棋子图片进行再处理，送入模型；
4. 棋子移动识别：采用前后图片的红色像素值对比，判断移动情况；
5. 标准棋局摆放：输入标准fen字符串，生成路径进行摆放；
6. 残局摆放：输入fen字符串，多余的棋子会拿走，然后和标准棋局摆放同理；
7. 滑轨的驱动：采用grbl操作系统，基于cnc加工的滑轨进行改造，通过串口发送命令，用esp32控制4个步进电机驱动；
8. 电磁铁的驱动：考虑到电磁铁的磁滞曲线，有剩磁，用l298n进行驱动，需要3个io口进行控制。


## 运行流程
1. 运行[recognize_board.py](recognize_board.py)首先将识别棋盘函数的is_find_4_corner改为False，拖动滑块，保证角点都在矩形的四个角就行，按q退出；
2. 之后is_find_4_corner改为True，理论上棋盘棋盘格点会显示出来，按q退出；
3. 之后运行[test_circle.py](test_circle.py)，调解滑块，找到圆稳定为止，拖动颜色阈值滑块，确保红色棋子和黑色棋子框出的颜色不一样，按q退出；
4. 运行[camera.py](camera.py)，点击图片，第一次点击为左上角，第二次点击为右下角，按q退出，划出感兴趣的区域；
5. 运行[cheeInterface.py](cheeInterface.py)，里面有摆放开局和正常识别开局，摆放开局需要输入fen字符串，正常识别开局是识别用户提前摆放的局面，只需要改几个变量就行，理论上可以识别任意局面，注意红色和
  黑色棋子方向，以及摄像头拍摄棋盘的方向，还有摆放开局的时候棋子必须在中央，可以参考我拍摄的图片，在[output](output)里面；
6. 识别完之后就是串口初始化和电磁铁的初始化，由于电磁铁我用的是树莓派4b控制的，所以采用了socket通讯，理论上用个单片机就行，没必要用树莓派；
7. 会弹出一个界面，用户选择执红还是执黑；
8. 用户走完棋子按下回车程序会继续运行。

# 注意事项
1. 训练权重文件没有上传，因为太大了，而且我的棋子比较小，每个人棋子可能不一样，需要重新训练；
2. 我的权重链接：https://pan.baidu.com/s/18oXEkN-A8yhUaJFq22WGbw?pwd=i5dt ；
3. 鉴于本人水平有限，程序或多或少有些bug，请大家一起交流，共同改进；
4. 我的棋盘是A3大小，20cm长度，在项目中已经提供，如果棋盘不一样，需要自行修改参数；
5. [ArmIK](ArmIK)是我研究机械臂时的一些算法，用的是几何算法，仅供参考，和滑轨没有关系；
6. 残局和标准棋局摆放请尽量把棋子放到中间，也就是竖着的棋盘上下各空出来三行，感兴趣的可以自己设计摆棋算法，我的算法仅供参考；
7. 由于采用的是通过判断前后图片识别用户走法，因此程序设计成了在滑轨归位之后进行拍照保存，所以需要等待机器归位后大约不到0.5s才能继续走棋，
同时也增加了容错性处理，机器会拍摄滑轨归位前三帧和后五帧，如果均判断失败，会重新检测局面，只不过花费的时间稍长些罢了。

## 程序不足
鉴于时间有限和个人水平有限，还是存在些许不足
1. 因为选用的是分类检测模型，局面检测最多处理32个棋子，也就是32张图片，采用的是顺序执行，理论上多张图片并行推理会更快；
2. 数据集太少，目前每种棋子的采样大概不到1000张，个别极端光照情况下和各种情况的数据集可能依旧没有采集到，偶尔会有误识别情况发生；
3. 程序有些地方写得过于臃肿和繁琐，在高性能x86的pc机上感受不到速度，但是放到嵌入式设备上速度就明显感受到了，而且python这种解释性语言本来就慢；
4. 在手眼标定方面采用的是矩形逆变换的方式，因此对于棋盘的摆放要求较高，滑轨的末端执行器必须保证在棋盘的框上能运动出标准的矩形，这样才能保证抓取时的误差最小；
5. 在设备安全性上没有做处理，当使用者将手伸入导轨时，如果导轨正在移动，可能会有安全隐患；
6. 摆棋子的算法目前采用的是先摆两边，后摆中间的思路，而且也考虑到了棋子摆放时候的占位碰撞可能性，但是没有考虑滑轨移动完每个棋子时候的位置，因此路径并不是最短的，可以继续优化；
7. 因为象棋界面是移植他人的程序，没搞懂线程交互问题，有时走完棋按下回车没反应，需要再次按下回车。

## 演示视频
https://www.bilibili.com/video/BV1em42157BR

# 致谢
感谢各位无私的大佬开源，下面是给我提供参考的链接
1. 皮卡鱼引擎
  https://github.com/official-pikafish/Pikafish ；
2. 界面设计
  https://github.com/StevenBaby/chess ；
  https://github.com/windshadow233/python-chinese-chess ；
3. 胜负判断
  https://github.com/bupticybee/XQPy ；
4. 模型的训练
  https://github.com/chenyr0021/Chinese_character_recognition/tree/master ；
5. 思路启蒙
  https://github.com/STM32xxx/Chinese-chess-robot-upper-computer ；
