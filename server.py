import logging
import os
from tempfile import NamedTemporaryFile
import requests
import hydra
import torch
from flask import Flask, request, jsonify
from hydra.core.config_store import ConfigStore
import time
from deepspeech_pytorch.configs.inference_config import ServerConfig
from deepspeech_pytorch.inference import run_transcribe
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.utils import load_model, load_decoder
from flask import send_file, send_from_directory
from flask import render_template
from flask_cors import CORS, cross_origin
from flaskext.mysql import MySQL
from ConvertAudioToWav import ConvertAudioToWav
import glob
import time
import datetime
import json
import shutil
import moviepy.editor as mp
from Punction import transcribe_comma
mysqlService = False
mysql = None
conn = None
app = Flask(__name__)

import Levenshtein as Lev

def wer(s1, s2):
    b = set(s1.strip().split() + s2.strip().split())
    word2char = dict(zip(b, range(len(b))))
    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    return Lev.distance(''.join(w1), ''.join(w2))   

def cer( target, hypothesis):
    target, hypothesis, = target.replace(' ', ''), hypothesis.replace(' ', '')
    return Lev.distance(target, hypothesis)
import re
def werPecentage(target, hypothesis):
    hypothesis = ''.join(re.split(r'[.;!?,\n,\t\,\r]', hypothesis)).strip().lower()
    target = ''.join(re.split(r'[.;!?,\n,\t\,\r]', target)).strip().lower()
    num=wer(target, hypothesis)
    return (float(num)/len(target.split()))*100

def cerPecentage(target, hypothesis):
    hypothesis = ''.join(re.split(r'[.;!?,\n,\t\,\r]', hypothesis)).strip().lower()
    target = ''.join(re.split(r'[.;!?,\n,\t\,\r]', target)).strip().lower()
    num=cer(target, hypothesis)
    return (float(num)/len(target.replace(' ', '')))*100
try:
    mysql = MySQL()
    app.config['MYSQL_DATABASE_USER'] = 'root'
    app.config['MYSQL_DATABASE_PASSWORD'] = '07061999'
    app.config['MYSQL_DATABASE_DB'] = 'vnsr'
    app.config['MYSQL_DATABASE_Host'] = 'localhost'
    mysql.init_app(app)
    conn = mysql.connect()
    mysqlService = True
except Exception as rr:
    print(rr)
    mysqlService = False
def queryNonDataSql(sql):
    if mysqlService == False:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        return True
    except:
        return False
def queryDataSql(sql):
    if mysqlService == False:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.commit()
        return data
    except:
        return False
def download_file(url, folder_name):
    local_filename = url.split('/')[-1]
    path = os.path.join("/{}/{}".format(folder_name, local_filename))
    with requests.get(url, stream=True) as r:
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return path
ALLOWED_EXTENSIONS = set(['.wav', '.mp3', '.ogg', '.webm'])

cs = ConfigStore.instance()
cs.store(name="config", node=ServerConfig)

@app.errorhandler(404)
def page_not_found(e):
    request.path = request.path.replace("//", "/")
    if ("/download/" in request.path):
        req = request.path.split("/download/")
        status = True
        try:
            if (req[0] != '' and len(req < 2)):
                status = False
        except:
                status = False
        if status == False:
            return {'code': 403, 'msg': "Invalid"}
        filename = req[len(req)-1]
        path  = "/".join(req)
        path = path.replace("//", "/")
        return send_file(path, attachment_filename = filename, as_attachment=True)
    else:
        req = request.path.split("/file/")
        status = True
        try:
            if (req[0] != '' and len(req < 2)):
                status = False
        except:
                status = False
        
        if status == False:
            return {'code': 403, 'msg': "Invalid"}
        path  = "/".join(req)
        path = path.replace("//", "/")
        res = {}
        res['code'] = 200
        res['message'] = "index"
        files_path = glob.glob(path+"/*")
        if (path==None or len(files_path) < 1):
                res['code'] = 404
                res['message'] = 'No file / folder found'
                return res
        data = []
        for child in os.scandir(path):
            item = {}
            item['path'] = child.path
            item['name'] = child.name
            item['size'] = str( "%.1f" % (child.stat().st_size/ 1024))
            item['date_access'] = datetime.datetime.fromtimestamp(child.stat().st_atime).strftime('%Y/%m/%d %H:%M:%S')
            item['date_create'] = datetime.datetime.fromtimestamp(child.stat().st_mtime).strftime('%Y/%m/%d %H:%M:%S')
            if (os.path.isfile(child.path)):
                item['type'] = 'File'
            else:
                item['type'] = 'Folder'
            data.append(item)
        res['message'] = data
        return render_template('file.html', data = res)
@app.route('/transcribe', methods=['POST'])
@cross_origin()
def transcribe_file():
    if request.method == 'POST':
        res = {}
        res['total'] = 0
        res['seconds'] = 0
        t0 = time.time()
        transTxt = ""
        if 'file' not in request.files and 'url' not in request.form:
            res['code'] = 403
            res['data'] = "Missed audio files or url of mp3 file."
            return jsonify(res)
        try:
            #đây là trường hợp có 1 trong 2 tham số 'file' và 'url', hoặc có cả 2

            #***TH1 có tham số 'file'
            file_extension=""
            path=""#đường dẫn lưu file âm thanh ở server cần nhận dạng
            if('file' in request.files):
                file = request.files['file']
                filename = file.filename
                _, file_extension = os.path.splitext(filename)
                if file_extension.lower() not in ALLOWED_EXTENSIONS:
                    res['code'] = 403
                    res['data'] = "{} is not supported format.".format(file_extension)
                    return jsonify(res)
                with NamedTemporaryFile(prefix="product_",suffix=file_extension, dir='/work/dataset_product/wav', delete=False) as temp_audio:
                    file.save(temp_audio.name)#lưu file cần nhận dạng vào đường dẫn temp_audio.name
                    path = temp_audio.name
            elif ('url' in request.form):#***TH2 ko có 'file', có tham số 'url'
                url = request.form['url']#đường dẫn mp3 hoặc video

                #tải về
                folder='work/dataset_recording/wav'
                absolute_path=download_file(url, folder)

                #nếu là mp3
                if (absolute_path).endswith('.mp3'):
                    path=absolute_path

                #nếu là mp4 : tách audio trong video
                elif(absolute_path).endswith('.mp4'):
                    my_clip = mp.VideoFileClip(absolute_path)
                    filename=os.path.splitext(absolute_path)[0]
                    path=os.path.join(folder, filename+'.mp3')
                    my_clip.audio.write_audiofile(path)

                #nếu ko phải
                else:
                    res['code'] = 403
                    res['data'] = "Extension is not supported."
                    return jsonify(res)
                

            
            #Chuyển đổi file âm thanh đúng định dạng wav, trả về đường dẫn wav sau chuyển đổi, file wav mới có tên giống file cũ
            path = ConvertAudioToWav(path)

            print("File name : "+str(path))
            # strCovert = "ffmpeg -i "+"/transcribe_tmp/tmpbh97i2v0.webm" +" -c:a pcm_f32le "+/transcribe_tmp/ou2t.wav"
            choose = 1
            try:
                choose = int(request.form['model'])
            except:
                pass

            global model, model2, model3
            runingModel = model 
            if (choose==2):
                runingModel = model2
                print("Using model 2")
            if (choose==3):
                runingModel = model3
                print("Using model 3")
            transcription, transcriptionGreedy,_,_ = run_transcribe(audio_path=path,
                                            spect_parser=spect_parser,
                                            model=runingModel,
                                            decoder=decoder,
                                            device=device,
                                            use_half=True)
            res['status'] = 200
            res_text = ""
            if (len(transcription) > 0):
                res_text = transcription[0][0]
                res['total'] = len(transcription[0])
            else:
                res_text = transcription
                res['total'] = len(transcription)
            
            res['data'] = transcribe_comma.runTranscribe(commo_model,dict_data,  word_dict, char_dict, res_text)
            res['path'] = path
            res['greedy']=transcribe_comma.runTranscribe(commo_model,dict_data,  word_dict, char_dict, transcriptionGreedy[0][0])
            transTxt = path.replace("wav", "txt")
            with open(transTxt,"w") as textFile:
                textFile.write(res['data'])
            logging.info('Success transcript')
            logging.debug(res)
        except Exception as exx:
            res['status'] = 403
            res['data'] = "Không thể nhận dạng\n"+str(exx)
        t1 = time.time()
        total = t1-t0
        targetString = ""
        wer = 100
        cer = 0
        try:
            targetString = request.form['targetString']
            wer = werPecentage(targetString, res_text)
            cer = cerPecentage(targetString, res_text)
        except:
            wer = 0
            er = 0
        res['seconds'] = total
        res['wer'] = round(wer, 3)
        res['cer']= round(cer, 3)
        return res
# 
# Get transcribe FPT
@app.route('/fpt', methods=['POST'])
@cross_origin()
def FPTapi():
    path = None
    try:
        path = request.form['path']
    except:
        pass

    data = queryDataSql('SELECT * FROM apikey LIMIT 1')
    if (data == False):
        return json.dumps({"status": 403, "msg": "Không thể truy cập database"}, ensure_ascii=False)
    if  (len(data) == 0):
        return json.dumps({"status": 404, "msg": "Đã hết key FPT"}, ensure_ascii=False)
    key = data[0][2]
    keyid = data[0][0]
    try:
        url = 'https://api.fpt.ai/hmi/asr/general'
        payload = None
        if (path != None):
            payload = open(path, 'rb').read()
        else:
            return json.dumps({"status": 404, "msg": "{0}".format(str("V1 API, not allow null path"))}, ensure_ascii=False)
        headers = {
            'api-key': '{0}'.format(key)
        }
        response = requests.post(url=url, data=payload, headers=headers)
        result = response.json()
        if (response.status_code == 401):
            print(result['message'])
            queryRes = queryNonDataSql('DELETE FROM apikey WHERE apikey.id = {0}'.format(keyid))
            FPTapi()
        if (response.status_code != 200):
            return json.dumps({"status": 404, "msg": "{0}".format(str("API not found"))}, ensure_ascii=False)
        if (result['status'] != 0):
            return json.dumps({"status": 403, "msg": "{0}".format(result['message'])}, ensure_ascii=False) 
        trans = result['hypotheses']
        if (len(trans) < 0):
            return json.dumps({"status": 403, "msg": "{0}".format("Không tìm thấy kết quả")}, ensure_ascii=False)
        return  json.dumps({"status": 200, "msg": "{0}".format(trans[0]['utterance'])}, ensure_ascii=False)
    except Exception as err:
        return json.dumps({"status": 403, "msg": "{0}".format(str(err))}, ensure_ascii=False)

# @app.route('/suggest', methods=['POST'])
# @cross_origin()
# def suggest_file():
#     if request.method == 'POST':
#         res = {}
#         res['total'] = 0
#         transTxt = ""
#         if 'file' not in request.files:
#             res['code'] = 403
#             res['data'] = "Missed audio files"
#             return jsonify(res)
#         try:
#             file = request.files['file']
#             filename = file.filename
#             _, file_extension = os.path.splitext(filename)
#             if file_extension.lower() not in ALLOWED_EXTENSIONS:
#                 res['code'] = 403
#                 res['data'] = "{} is not supported format.".format(file_extension)
#                 print(res['data'])
#                 return jsonify(res)
#             with NamedTemporaryFile(prefix="product_",suffix=file_extension, dir='/work/dataset_fpt/wav', delete=False) as temp_audio:
#                 file.save(temp_audio.name)
#                 path = temp_audio.name
#                 if (file_extension.lower()==".webm"):
#                     src1 = temp_audio.name #.webm
#                     dst1 = temp_audio.name #.webm
#                     dst1 = dst1.replace("webm", "mp3") #.mp3
#                     convertWebmToMp3(src1, dst1)  #.wav
#                     src2=dst1
#                     dst2=dst1.replace("mp3", "wav")
#                     convertMp3ToWav16(src2, dst2)
#                     os.remove(dst1)
#                     try:
#                         os.remove(src1)
#                     except:
#                         pass
#                     os.remove(dst1)
#                     path = dst2
#                 if (file_extension.lower()==".mp3"):
#                     src= temp_audio.name
#                     dst=src.replace("mp3", "wav")
#                     convertMp3ToWav16(src, dst)
#                     path = dst
#                 print("File name : "+str(path))
#                 transcription, _ = run_transcribe(audio_path=path,spect_parser=spect_parser,model=model,decoder=decoder,device=device,use_half=True)
#                 res['status'] = 200
#                 if (len(transcription) > 0):
#                     res['data'] = transcription[0][0]
#                     res['total'] = len(transcription[0])
#                 else:
#                     res['data'] = transcription
#                     res['total'] = len(transcription)
#                 transTxt = path.replace("wav", "txt")
#                 with open(transTxt,"w") as textFile:
#                     textFile.write(res['data'])
#                 #os.remove(dst2)
#         except Exception as exx:
#             res['status'] = 403
#             res['data'] = str(exx)
#         return res
# 
@app.route('/file')
def index(name):
    res = {}
    res['code'] = 200
    res['message'] = "index"
    files_path = glob.glob(name+"/*")
    if (name==None or len(files_path) < 1):
            res['code'] = 404
            res['message'] = 'No file / folder found'
            return res
    data = []
    for child in os.scandir("/work"):
        item = {}
        if (os.path.isfile(child.path)):
            item['path'] = child.path
            item['name'] = child.name
            item['size'] = str( "%.1f" % (child.stat().st_size/ 1024))
            item['date_access'] = datetime.datetime.fromtimestamp(item.stat().st_atime).strftime('%Y/%m/%d %H:%M:%S')
            item['date_create'] = datetime.datetime.fromtimestamp(item.stat().st_mtime).strftime('%Y/%m/%d %H:%M:%S')
            item['date_create'] = datetime.datetime.fromtimestamp(item.stat().st_mtime).strftime('%Y/%m/%d %H:%M:%S')
            item['type'] = 'File'
        else:
            item['path'] = child.path
            item['type'] = 'Folder'
        data.append(item)
    res['message'] = json.loads(data)
    return render_template('index.html')
@hydra.main(config_name="config")
def main(cfg: ServerConfig):
    global model, spect_parser, decoder, config, device, model2, model3
    global commo_model,dict_data,  word_dict, char_dict
    commo_model,dict_data,  word_dict, char_dict = transcribe_comma.loadModel()
    config = cfg
    model1Path = '/work/Source/deepspeech.pytorch/models/deepspeech_50_1600_gru_fpt.pth'
    logging.info('Setting up server...')
    device = torch.device("cuda" if cfg.model.cuda else "cpu")
    model = load_model(device=device,
                       model_path=model1Path,
                       use_half=cfg.model.use_half)
    logging.info('Loaded model 1')
    model2Path = '/work/Source/deepspeech.pytorch/models/deepspeech_1600_lstm_16_50_vin.pth'
    model2 = load_model(device=device,
                       model_path=model2Path,
                       use_half=cfg.model.use_half)

    logging.info('Loaded model 2')
    model3Path = "/work/Source/deepspeech.pytorch/models/deepspeech_checkpoint_epoch_43.pth"
    model3 = load_model(device=device,
                       model_path=model3Path,
                       use_half=cfg.model.use_half)
    logging.info('Loaded model 3')
    decoder = load_decoder(labels=model.labels,cfg=cfg.lm)
    spect_parser = SpectrogramParser(audio_conf=model.audio_conf,normalize=True)
    spect_parser = SpectrogramParser(model.audio_conf, normalize=True)
    logging.info('Server initialised')
    app.run(host=cfg.host, port=cfg.port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
