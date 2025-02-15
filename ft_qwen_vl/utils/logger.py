import logging
import os
import sys

class CustomLogger(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            if os.environ.get("GLOG_DEBUG", "False").lower() == "true":
                level = logging.DEBUG
            else:
                level = logging.INFO

            formatter = logging.Formatter( # [%(module)s]
                "[RM][%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]"
                "[pid:%(process)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(formatter)
            cls.instance = logging.getLogger("Qwen2_5_VL")
            cls.instance.addHandler(handler)
            cls.instance.setLevel(level)
            cls.instance.propagate = False
        return cls.instance


def get_custom_logger():
    return CustomLogger()
CUSTOM_LOGGER = get_custom_logger()
