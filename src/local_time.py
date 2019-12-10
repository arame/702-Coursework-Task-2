import time

class Localtime:
  @staticmethod
  def get():
    return time.asctime( time.localtime(time.time()) )