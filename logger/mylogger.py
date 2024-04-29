import datetime
import logging

import pytz

# 한국시간으로 날짜 포멧팅
class Formatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""

    def converter(self, timestamp):
        # Create datetime in UTC
        dt = datetime.datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        # Change datetime's timezone
        return dt.astimezone(pytz.timezone("Asia/Seoul"))

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="milliseconds")
            except TypeError:
                s = dt.isoformat()
        return s

# 로거 셋팅
def set_logger(file_name, version):
    """
    summary
        - 로깅 객체 및 포맷 정의
    """
    file_name = "./logs/" + file_name + '_' + version + ".txt"
    logging.basicConfig(filename=file_name, level=logging.INFO)
    
    
    mylogger = logging.getLogger("process")
    mylogger.setLevel(logging.INFO)
    mylogger.propagate = False

    formatter = Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    
    fh = logging.FileHandler(filename=file_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    mylogger.addHandler(fh)
    mylogger.addHandler(stream_handler)

    return mylogger