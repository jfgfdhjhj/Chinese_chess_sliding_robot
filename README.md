# Chinese chess robot, based on writing robot python implementation

Using pickup fish engine, electric slide xyz structure driven Chinese chess robot

## general process
1. Checkerboard search: call opencv's search corner function to save the checkerboard corner coordinates;
2. Situation initialization: Detect chess pieces with Hough circle detection, cut them and send them to model detection;
3. Chess piece identification: the data set classification model similar to minist is adopted to reprocess the original chess piece pictures and feed them into the model;
4. Chess piece movement recognition: use the red pixel value comparison of the front and back pictures to judge the movement;
5. Standard chess layout: Input standard fen string to generate a path for placing;
6. End place: Enter fen string, excess pieces will be taken away, and then the standard chess place the same;
7. Drive of the slide rail: grbl operating system is adopted, and the slide rail is transformed based on cnc machining. Commands are sent through serial port, and four stepper motors are controlled by esp32.
8. Electromagnet drive: Considering the hysteresis curve of the electromagnet, there is remanence, driven by ln298, which requires 3 io ports for control.


## running process
1. Run [recognize_board.py](recognize_board.py) to first change is_find_4_corner, which recognize_board.py recognizes the checkerboard function, to False, drag the slider to ensure that all corners are in the four corners of the rectangle, and press q to exit;
2. After is_find_4_corner is changed to True, theoretically the checkerboard grid points will be displayed, press q to exit;
3. Then run [test_circle.py](test_circle.py), adjust the slider until the circle is stable, drag the color threshold slider to make sure that the color of the red and black pieces is different, press q to exit;
4. Run [camera.py](camera.py), click the picture, click the top left corner for the first time, click the bottom right corner for the second time, press q to exit, and mark the area of interest;
5. Run [cheeInterface.py](cheeInterface.py), which has placed the opening and normal identification of the opening, placing the opening needs to input fen string, normal identification of the opening is to identify the user in advance of the situation, only need to change a few variables on the line, in theory can identify any situation, pay attention to red and
The direction of the black chess pieces, the direction of the chessboard shot by the camera, and the chess pieces must be in the center when placing the opening, you can refer to the picture I shot in [output](output);
6. After the identification is the serial port initialization and electromagnet initialization, because the electromagnet I use is Raspberry PI 4b control, so the use of socket communication, theoretically with a single chip computer on the line, there is no need to use raspberry PI;
7. An interface will pop up, and the user can choose red or black;
8. After the user finishes the chess piece, press Enter and the program will continue to run.

# matters need attention
1. The training weight file is not uploaded, because it is too big, and my chess pieces are relatively small, each person's chess pieces may be different, need to be retrained;
2. The weight of my links: https://pan.baidu.com/s/18oXEkN-A8yhUaJFq22WGbw?pwd=i5dt;
3. In view of my limited level, the program has more or less bugs, please communicate together and improve together;
4. My chessboard is A3 size and 20cm length, which has been provided in the project. If the chessboard is different, you need to modify the parameters by yourself;
5. [ArmIK](ArmIK) is some of the algorithms when I study the robot arm, using geometric algorithms for reference only, and has nothing to do with the slide;
6. The end game and standard chess layout, please try to put the chess pieces in the middle, that is, the vertical board above and below each empty three lines, interested can design their own algorithm, my algorithm is for reference only;
7. Because the user's moves are identified by judging the pictures before and after, the program is designed to take photos and save them after the slide return, so it is necessary to wait for about 0.5s after the machine returns to continue to move.
At the same time, it also increases the fault tolerance processing, the machine will shoot the first three frames and the last five frames of the slide return, if the judgment fails, it will re-detect the situation, but it takes a little longer.

## underprogram
Given the limited time and limited personal level, there are still some shortcomings
1. Because the classification detection model is selected, the situation detection processes a maximum of 32 chess pieces, that is, 32 pictures, and adopts sequential execution. Theoretically, parallel reasoning with multiple pictures will be faster;
2. There are too few data sets, the sampling of each chess piece is less than 1000 pieces at present, and the data sets of individual extreme lighting conditions and various situations may still not be collected, and occasionally wrong identification will occur;
3. Some parts of the program are too bloated and cumbersome, and can not feel the speed on high-performance x86 PCS, but the speed on embedded devices is obviously felt, and python is a slow interpreted language;
4. The hand-eye calibration adopts the mode of inverse rectangular transformation, so the requirements for the placement of the chessboard are high. The end effector of the slide rail must ensure that the standard rectangle can be moved on the chessboard frame, so as to ensure the minimum error when grabbing;
5. The safety of the equipment is not handled. When the user reaches into the guide rail, if the guide rail is moving, there may be security risks;
6. At present, the algorithm of placing chess pieces adopts the idea of first placing both sides and then placing the middle, and also takes into account the possibility of occupying collision when the chess pieces are placed, but does not consider the position of each chess piece after moving the slide rail, so the path is not the shortest and can be further optimized;
7. Because the chess interface is a program transplanted to others, I did not understand the thread interaction problem, and sometimes the press enter did not respond after playing chess, and I need to press enter again.

## demonstration video
https://www.bilibili.com/video/BV1em42157BR

# thanks
Thanks to the selfless big guys open source, the following is to provide me with a reference link
1. Pickup fish engine
  https://github.com/official-pikafish/Pikafish ；
2. interfacial design
  https://github.com/StevenBaby/chess ；
  https://github.com/windshadow233/python-chinese-chess ；
3. Win or lose judgment
  https://github.com/bupticybee/XQPy ；
4. Model training
  https://github.com/chenyr0021/Chinese_character_recognition/tree/master ；
5. Enlightenment of thought
  https://github.com/STM32xxx/Chinese-chess-robot-upper-computer ；
