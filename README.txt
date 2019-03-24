Autosub_Multiple_Methods
Autosub based on multiple methods which are deep moedls and web API 

综述：
这个项目的主要目的是实现视频文件字幕的自动生成，基于开源项目autosub(https://github.com/agermanidis/autosub)
整合完成。目前，实现的最好的效果是基于百度API，事实上，作者还尝试了几个基于深度学习模型的方案，测试效果很不理想。这些方案均位于other_methods。

baidusub3:该文件夹内的代码是基于autosub和百度的语言转换API接口完成的，效果还不错。适合国内学习和使用。详细情况参见文件夹内的README.

other_methods:
	1，autosub2_vpn:这个是autosub的源码，基于Python2，但是可能和官方现在的版本稍有不同。autosub2_vpn做了相应修改可以载win上直接使用命令行执
	行py文件来执行程序。但是你需要提前装好依赖，方法参见该文件夹下的README.
	2，autosub3_vpn:也是autosub的另一个分支源码，基于python3，只是python版本不同。使用方式同autosub2_vpn，先安装好依赖包，然后使用命令行执行
	这两个版本都是autosub的官方源码，需要接入Google API，感谢原作者贡献了Google API KEY,可供测试，但是需要科学上网，
	否则没有任何结果。科学上网方法请自行解决。
	
	3，autosub3_xfyun：这个方法是基于讯飞的语言听写API接口实现的，但是讯飞的接口并不是免费的，测试期间每个用户每天只能申请500次，而且每个接口只
	运行最多5个IP访问，需要在讯飞官网设置，并不实用。有需要的可以做参考。实用方式：按照README安装依赖，命令行运行即可。Python3版本。
	
	4，autosub3_model：次版本是基于autosub的python3版本，结合开源Deepspeech（https://github.com/audier/DeepSpeechRecognition）
	实现的，由于该项目并没有发布训练好的模型，作者本人使用项目提供的数据集做了相关训练，但是效果并不好，若有想要继续深入研究的人，欢迎参考。实用
	方式查看README.
	
	5，baidusub2_model：此版本是基于autosub的python2版本，结合开源项目Deepspeech2（https://github.com/PaddlePaddle/DeepSpeech）
	实现的。Deepspeech2是基于百度的深度学习框架paddle实现的百度开源deepspeech方案。此方案使用官方发布的aishell数据训练过的模型。效果比上一个
	深度模型方案好，但是也不尽如人意。使用方式详见README.
	注意：如果运行中有解决不了的问题，可以尝试怀疑是模型损坏，
	
	6，web_api_demo：
	是百度和讯飞的语言API示例代码。和官网的一致，也可直接从官网下载。

注：
1，所有方法均须安装好ffmpeg，linux下直接使用sudo apt-get install ffmpeg即可，windows下可以下载编译好的ffmpeg.exe文件
加入系统环境变量即可，测试方法:在命令行输入"ffmpeg -h"可以打印帮助信息表示ffmpeg安装成功。

2，当前代码均是在linux版本下运行，强烈建议在linux下运行，若需要在windows上运行版本1,2,3,4需要将其中的主要文件autosub_xx.py中的代码做如下修改：
<1>:
```
if not os.path.isfile(filepath):
        print("The given file does not exist: {}".format(filepath))
        raise Exception("Invalid filepath: {}".format(filepath))
    if not which("ffmpeg"):
        print("ffmpeg: Executable not found on machine.")
        raise Exception("Dependency not found: ffmpeg")
```
		
	以上代码删除
<2>：use_shell = True if os.name == "nt" else False
    subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
	以上代码修改为：
	subprocess.check_output(command)

<3>：在win上运行注意路径用'\\'，在linux上用'/'
使用中遇到任何问题请联系作者568884899@qq.com


