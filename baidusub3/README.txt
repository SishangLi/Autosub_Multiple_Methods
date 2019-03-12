环境
	1，平台：linux
	2，python3.6
	3，依赖包安装方法：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt


使用说明：
	----Linux
	1，源码放在/home/share/lmcproject/Task_one/autosub，该路径下的不同文件夹为任务一的不同版本，
	   本文档只讲述baidusub3的使用方法。
	2，在linux下的使用方式可以参照Linux_API.py   主要有一下几个要点：
		第一：需要使用sys.path.append()将源码路径加入python解释器路径
		第二：使用 import baidusub3.main as autosub 引入模块
		第三：需要制定需要生成字幕的视频的路径（相对路径、绝对路径均可）
		第四：使用result = autosub.start(videopath=videopath)来执行
		第五：autosub.start(videopath=videopath)有其他选填参数，详细信息可以参见baidusub3的main.py中函数start()中的注释	
		第六：函数返回值result 为字幕文件绝对路径

注：
	1，生成过程会在当前Linux_API.py所在路径生成‘temp’文件夹，用来存放中间文件，程序程序正常返回时，该文件夹会自动删除，但是当程序异常终止，下次继续执行时，需要手动删除‘temp’文件夹。
	2，生成的字幕文件xxx.srt，在播放器中载入时可能会出现乱码的情况，但是手动打开查看没有这种情况。此bug原因尚未查明，十分抱歉。不过应该不影响后续任务对接，若后续任务读取.srt文件后有异常，请及时联系我们。
	






