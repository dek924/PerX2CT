import datetime
from pytz import timezone
from pathlib import Path

def add_3D_image(tensor, max_image):
  '''
  :param tensor:
    (c,h,w)
  :param max_image:
  :return:
  '''
  c, h, w = tensor.size()
  if c <= max_image:
    images = [tensor[i:i+1] for i in range(c)]
  else:
    skip_len = float(c) / max_image
    images = [tensor[int(i*skip_len):(int(i*skip_len) + 1)] for i in range(max_image)]

  return images


def tensor_back_to_unnormalization(input_image, mean, std):
  image = input_image * std + mean
  return image


class Log:
    def __init__(self, root, is_print=True):
        self.root = root
        self.is_print = is_print

    def write(self, string, end='\n'):
        datetime_string = datetime.datetime.now(timezone('Asia/Seoul')).strftime("%d-%m-%y %H:%M:%S")
        string = '%s: %s' % (datetime_string, string)
        if self.is_print:
            print(string, end=end)
        with open(self.root, 'a') as f:
            f.write(string + end)

class ExperimentLogger:
    def __init__(self, save_root, log_file='log.log', is_print=True):
        self.save_root = save_root
        Path(self.save_root).mkdir(parents=True, exist_ok=True)
        self.log = Log(f"{self.save_root}/{log_file}", is_print=is_print)

    def log_loss(self, loss_name, loss):
        self.log.write(f'{loss_name}: {loss:.4f}')