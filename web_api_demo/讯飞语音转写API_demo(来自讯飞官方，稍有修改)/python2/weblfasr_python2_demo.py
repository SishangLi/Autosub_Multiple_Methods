# -*- coding: utf-8 -*-
 
import sys
import hashlib
from hashlib import sha1
import hmac
import base64
import json, time
import httplib, urllib
import os
import random 

reload(sys)
sys.setdefaultencoding('ISO-8859-1')

lfasr_host = 'raasr.xfyun.cn'
# 讯飞开放平台的appid和secret_key
app_id = ''
secret_key = ''
# 请求的接口名
api_prepare = '/prepare'
api_upload = '/upload'
api_merge = '/merge'
api_get_progress = '/getProgress'
api_get_result = '/getResult'
# 文件分片大下52k
file_piece_sice = 10485760
# 要是转写的文件路径
uplaod_file_path = ''

base_header = {'Content-type': 'application/x-www-form-urlencoded',  'Accept': 'application/json;charset=utf-8'}

# ——————————————————转写可配置参数————————————————
# 转写类型
lfasr_type = 0
# 是否开启分词
has_participle = 'false'
# 多候选词个数
max_alternatives = 0
# 子用户标识
suid = ''

def prepare():
    return lfasr_post(api_prepare, urllib.urlencode(generate_request_param(api_prepare)), base_header)

def upload(taskid):
    file_object = open(uplaod_file_path, 'rb')
    try:
        index = 1
        sig = SliceIdGenerator()
        while True:
            content = file_object.read(file_piece_sice)
            if not content or len(content) == 0:
                break
            response = post_multipart_formdata(generate_request_param(api_upload, taskid, sig.getNextSliceId()), content)
            if json.loads(response).get('ok') != 0:
                # 上传分片失败
                print 'uplod slice fail, response: '+ response
                return False
            print 'uoload slice ' + str(index) + ' success'
            index += 1
    finally:
        'file index:' + str(file_object.tell())
        file_object.close()

    return True

def merge(taskid):
    return lfasr_post(api_merge, urllib.urlencode(generate_request_param(api_merge, taskid)), base_header)

def get_progress(taskid):
    return lfasr_post(api_get_progress, urllib.urlencode(generate_request_param(api_get_progress, taskid)), base_header)

def get_result(taskid):
    return lfasr_post(api_get_result, urllib.urlencode(generate_request_param(api_get_result, taskid)), base_header)

# 根据请求的api来生成请求参数
def generate_request_param(apiname, taskid = None, slice_id = None):
    # 生成签名与时间戳
    ts = str(int(time.time()))
    tmp = app_id + ts
    hl = hashlib.md5()
    hl.update(tmp.encode(encoding='utf-8'))
    signa = base64.b64encode(hmac.new(secret_key,  hl.hexdigest(), sha1).digest())

    param_dict = {}

    # 根据请求的api_name生成请求具体的请求参数
    if apiname == api_prepare:
        file_len = os.path.getsize(uplaod_file_path)
        parentpath, shotname, extension = get_file_msg(uplaod_file_path)
        file_name = shotname + extension
        temp1 = file_len / file_piece_sice
        slice_num = file_len / file_piece_sice + (0 if (file_len % file_piece_sice == 0) else 1)

        param_dict['app_id'] = app_id
        param_dict['signa'] = signa
        param_dict['ts'] = ts
        param_dict['file_len'] = str(file_len)
        param_dict['file_name'] = file_name
        param_dict['lfasr_type'] = str(lfasr_type)
        param_dict['slice_num'] = str(slice_num)
        param_dict['has_participle'] = has_participle
        param_dict['max_alternatives'] = str(max_alternatives)
        param_dict['suid'] = suid
    elif apiname == api_upload:
        param_dict['app_id'] = app_id
        param_dict['signa'] = signa
        param_dict['ts'] = ts
        param_dict['task_id'] = taskid
        param_dict['slice_id'] = slice_id
    elif apiname == api_merge:
        param_dict['app_id'] = app_id
        param_dict['signa'] = signa
        param_dict['ts'] = ts
        param_dict['task_id'] = taskid
        parentpath, shotname, extension = get_file_msg(uplaod_file_path)
        file_name = shotname + extension
        param_dict['file_name'] = file_name
    elif apiname == api_get_progress or apiname == api_get_result:
        param_dict['app_id'] = app_id
        param_dict['signa'] = signa
        param_dict['ts'] = ts
        param_dict['task_id'] = taskid
    return param_dict

def get_file_msg(filepath):
    (parentpath,tempfilename) = os.path.split(filepath);  
    (shotname,extension) = os.path.splitext(tempfilename);  
    return parentpath,shotname,extension

def lfasr_post(apiname, requestbody, header):
    conn = httplib.HTTPConnection(lfasr_host)
    conn.request('POST', '/api' + apiname, requestbody, header)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return data

def post_multipart_formdata(strparams, content):
    BOUNDARY = '----------%s' % ''.join(random.sample('0123456789abcdef', 15))
    multi_header = {'Content-type': 'multipart/form-data; boundary=%s' % BOUNDARY, 'Accept': 'application/json;charset=utf-8'}
    CRLF = '\r\n'
    L = []
    for key in strparams.keys():
        L.append('--' + BOUNDARY)
        L.append('Content-Disposition: form-data; name="%s"' % key)
        L.append('')
        L.append(strparams[key])

    L.append('--' + BOUNDARY)
    L.append('Content-Disposition: form-data; name="%s"; filename="%s"' % ('content', strparams.get('slice_id')))
    L.append('Content-Type: application/octet-stream')
    L.append('')
    L.append(content)
    L.append('--' + BOUNDARY + '--')
    L.append('')
    body = CRLF.join(L)

    data = lfasr_post(api_upload, body, multi_header)

    return data

class SliceIdGenerator:
    """slice id生成器"""
    def __init__(self):
        self.__ch = 'aaaaaaaaa`'

    def getNextSliceId(self):
        ch = self.__ch
        j = len(ch) - 1
        while j >= 0:
            cj = ch[j]
            if cj != 'z':
                ch = ch[:j] + chr(ord(cj) + 1) + ch[j+1:]
                break
            else:
                ch = ch[:j] + 'a' + ch[j+1:]
                j = j -1
        self.__ch = ch
        return self.__ch

def  request_lfasr_result():
    # 1.预处理
    pr = prepare()
    prepare_result = json.loads(pr)
    if prepare_result['ok'] != 0:
        print 'prepare error, ' + pr
        return

    taskid = prepare_result['data']
    print 'prepare success, taskid: ' + taskid

    # 2.分片上传文件
    if upload(taskid):
        print 'uplaod success'
    else :
        print 'uoload fail'

    # 3.文件合并
    mr = merge(taskid)
    merge_result = json.loads(mr)
    if merge_result['ok'] != 0:
        print 'merge fail, ' + mr
        return

    # 4.获取任务进度
    while True:
        # 每隔20秒获取一次任务进度
        progress = get_progress(taskid)
        progress_dic = json.loads(progress)
        if progress_dic['err_no'] != 0 and progress_dic['err_no'] != 26605:
            print 'task error: ' + progress_dic['failed']
            return
        else :
            data = progress_dic['data']
            task_status = json.loads(data)
            if task_status['status'] == 9:
                print 'task ' + taskid + ' finished'
                break
            print 'The task ' + taskid + ' is in processing, task status: ' + data

        # 每次获取进度间隔20S
        time.sleep(20)

    # 5.获取结果
    lfasr_result = json.loads(get_result(taskid))
    print "result: " + lfasr_result['data']

if __name__ == '__main__':
    request_lfasr_result()