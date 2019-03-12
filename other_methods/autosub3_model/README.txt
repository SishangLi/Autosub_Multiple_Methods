安装方法：
1，pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
2，conda install tensorflow>=1.8.0 keras>=2.2.4

注：
1，需要先建立python3.6的python虚拟环境
2，直接安上述安装方法安装依赖库
3，python autosub_model.py xxx.mp4 执行
4，python autosub_model.py --help 显示帮助信息
5，logs_am和logs_lm是声学和语言模型
6，model_prepare中的代码用来训练模型
7，vocabulary中的代码用来构建词库
（注意，vocabulary中构建的词库需要和训练中用到的词库词序相对应（即，制作词库用到的训练集顺序要一致），
   因为模型是转换为序号查询拼音和汉子的）


