import sys
sys.path.append('/home/share/lmcproject/Task_one/autosub/')
sys.path.append('/home/share/lmcproject/Task_one/autosub/baidusub3')
import baidusub3.main as autosub
videopath = '../video/CCTV_News.mp4'


if __name__ == '__main__':
    result = autosub.start(videopath=videopath)
    print(result)

