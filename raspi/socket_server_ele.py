#!/usr/bin/env python3
# encoding:utf-8
import time
import socket
import json
import threading
import RPi.GPIO as GPIO

pin_in1 = 20
pin_in2 = 16
pin_ena = 21

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin_in1, GPIO.OUT)
GPIO.setup(pin_in2, GPIO.OUT)
GPIO.setup(pin_ena, GPIO.OUT)
_pwm = GPIO.PWM(pin_ena, 100)  # 创建PWM0实例，并设置频率为100Hz
# 启动 PWM·
_pwm.start(0)  # 占空比为 0，电磁铁停止


def electromagnet_forward(duty_cycle=100.0):
    _pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    GPIO.output(pin_in1, GPIO.LOW)
    GPIO.output(pin_in2, GPIO.HIGH)
    print("电磁铁正转")


def electromagnet_stop():
    _pwm.ChangeDutyCycle(0)
    time.sleep(0.1)
    print("电磁铁断电")


def electromagnet_back(duty_cycle=100.0):
    _pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    GPIO.output(pin_in1, GPIO.HIGH)
    GPIO.output(pin_in2, GPIO.LOW)
    print("电磁铁反转")


def must_fall():
    duty_cycle = 1e-9
    electromagnet_back(duty_cycle)
    time.sleep(0.1)
    electromagnet_forward()
    time.sleep(1)
    electromagnet_back(duty_cycle)
    time.sleep(0.1)
    electromagnet_stop()


def suction_and_drop(t=1):
    electromagnet_forward()
    time.sleep(t)
    must_fall()


def electromagnet_cleanup():
    GPIO.cleanup()


class RobotServerEle:
    HOST_PORT = 8969
    BUFFER_SIZE = 1024

    blank = "blank"
    finish = "finish"
    bye = "bye"

    finish_json = json.dumps(finish)
    bye_json = json.dumps(bye)

    topic_list = [
        "attract", "fall", "bye"
    ]

    def __init__(self, is_vim_address=True):

        # # 创建命令队列
        # command_queue = queue.Queue()
        # self.command_queue = command_queue
        self.should_exit = False
        if is_vim_address:
            HOST_IP = "192.168.139.136"
        else:
            HOST_IP = '169.254.150.116'
        self.HOST_ADDRESS = (HOST_IP, self.HOST_PORT)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(self.HOST_ADDRESS)
        s.listen(1)
        self.s = s
        self.start_server()
        print("服务器启动成功")
        # thread1 = threading.Thread(target=self.receiving_thread, daemon=False)
        # thread1.start()

    def start_server(self):
        self.should_exit = False
        thread1 = threading.Thread(target=self.receiving_thread, daemon=False)
        thread1.start()

    def stop_server(self):
        self.should_exit = True

    def receiving_thread(self):
        while not self.should_exit:
            try:
                client_socket, address = self.s.accept()  # 被动接受TCP客户端连接，持续等待直到连接到达（阻塞等待）
                print("建立成功, 连接来自address: %s:%s" % (address[0], address[1]))
                while True:
                    recvd = client_socket.recv(self.BUFFER_SIZE).decode('utf-8')
                    try:
                        jsonlist = json.loads(recvd)
                        res = self.msg_parse(jsonlist)
                        client_socket.send(bytes(res, encoding='utf-8'))

                    except json.decoder.JSONDecodeError:
                        print("客户端%s断开连接" % address[0])
                        break

                    except KeyboardInterrupt:
                        electromagnet_cleanup()
                        print("服务器内层关闭")
                        break

                    except Exception as e:
                        print("内层发生了其他异常：", e)
                        electromagnet_cleanup()
                        print("服务器内层关闭")
                        break

            except ConnectionResetError:
                print("客户端%s断开连接" % address[0])
                electromagnet_stop()
                continue

            except KeyboardInterrupt:
                print("服务器关闭")
                electromagnet_cleanup()
                break

            except Exception as e:
                print("内层发生了其他异常：", e)
                electromagnet_cleanup()
                print("服务器关闭")
                break

        print("服务器已经关闭")

    # def check_connection(self):
    #     if not self.connected:
    #         print("客户端连接断开，重新启动服务器...")
    #         self.s.close()
    #         self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         self.s.bind(self.HOST_ADDRESS)
    #         self.s.listen(1)
    #         self.start_server()  # 重新启动服务器
    #     self.reconnect_timer = threading.Timer(5, self.check_connection)  # 重新启动定时器
    #     self.reconnect_timer.start()

    def msg_parse(self, msg):
        if msg[0] == self.topic_list[0]:
            electromagnet_forward()
            return self.finish_json

        elif msg[0] == self.topic_list[1]:
            electromagnet_back()
            electromagnet_stop()
            return self.finish_json

        elif msg[0] == self.topic_list[2]:
            return self.bye_json
        return False


if __name__ == "__main__":
    robot_server_ele = RobotServerEle(is_vim_address=False)
