import logging
import colorlog

log_colors_config = {
    'DEBUG': 'fg_bold_white',  # cyan white
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}
logger = logging.getLogger(__name__)
# 输出到控制台
console_handler = logging.StreamHandler()

# 日志级别，logger 和 handler以最高级别为准，不同handler之间可以不一样，不相互影响
# 设置打印日志的级别，level级别以上的日志会打印出
# level=logging.DEBUG 、INFO 、WARNING、ERROR、CRITICAL
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)
# console_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)


# 日志输出格式
console_formatter = colorlog.ColoredFormatter(
    fmt="%(log_color)s[%(levelname)s] %(log_color)s[%(asctime)s.%(msecs)03d]: %(message)s  -> "
        "%(filename)s: %(funcName)s "
        "line:%(lineno)d ",
    # datefmt='%Y-%m-%d  %H:%M:%S',
    datefmt='%H:%M:%S',
    log_colors=log_colors_config
)
console_handler.setFormatter(console_formatter)

# 重复日志问题：
# 1、防止多次addHandler；
# 2、loggername 保证每次添加的时候不一样；
# 3、显示完log之后调用removeHandler
if not logger.handlers:
    logger.addHandler(console_handler)


console_handler.close()

if __name__ == '__main__':
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')

