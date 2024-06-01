#!/usr/bin/env python3
# encoding:utf-8
import socket
import json
from chess_utils.logger import logger


class ClientEle:
    port = 8969

    blank = "blank"
    finish = "finish"

    topic_list = [
        "attract", "fall", "bye"
    ]

    # 创建客户端套接字
    def __init__(self, _is_vm=False):
        if _is_vm:
            _ip_address = "192.168.139.136"
        else:
            _ip_address = '192.168.1.116'
        c = socket.socket()
        self.c = c
        c.connect((_ip_address, self.port))
        print("与副服务器连接建立成功", _ip_address)

    def send_topic_msg(self, topic, *msg):
        send_list = [topic, *msg]
        send_json_list = json.dumps(send_list)
        logger.debug(send_json_list)
        self.c.send(bytes(send_json_list, encoding='utf-8'))
        res = self.c.recv(1024).decode('utf-8')
        # while True:
        #     try:
        #         jsonlist = json.loads(res)
        #         break
        #     except json.decoder.JSONDecodeError:
        #         continue
        jsonlist = json.loads(res)
        logger.debug(res)
        return jsonlist

    def send_electromagnet_attract(self):
        self.send_topic_msg(self.topic_list[1])

    def send_electromagnet_fall(self):
        self.send_topic_msg(self.topic_list[0])

    def send_bye(self):
        res = self.send_topic_msg(self.topic_list[2])
        return res

# def connect_to_server(host, port, sock):
#     while True:
#         try:
#             # 判断socket是否处于连接状态
#             if is_socket_connected(sock):
#                 time.sleep(2)
#                 pass
#             else:
#                 print("Socket is not connected")
#                 sock.connect((ip_address, port))
#                 print(f"Connected to {host}:{port}")
#         except Exception as e:
#             print(f"Failed to connect to {host}:{port}. Error: {e}")
#             # 连接失败后等待一段时间再尝试重新连接
#             time.sleep(2)


def is_socket_connected(sock):
    try:
        # 获取socket连接的远程地址
        sock.getpeername()
        return True
    except socket.error:
        return False


if __name__ == '__main__':
    clint_ele = ClientEle(_is_vm=False)
    while True:
        # 信息发送
        info = input('请输入命令,a吸取, f掉落:')
        if info == "a":
            clint_ele.send_electromagnet_attract()
        elif info == "f":
            clint_ele.send_electromagnet_fall()
