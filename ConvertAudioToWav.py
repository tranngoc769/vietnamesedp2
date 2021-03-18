import os
from convertWav16 import convertMp3ToWav16
from convertWebmToMp3 import convertWebmToMp3

def ConvertAudioToWav(path):
    _, file_extension = os.path.splitext(path)
    temp_audio_path=path
    if (file_extension.lower()==".webm"):
                #chuyen sang webm->mp3
        src1 = temp_audio_path #.webm
        dst1 = temp_audio_path #.webm
        dst1 = dst1.replace("webm", "mp3") #.mp3
        convertWebmToMp3(src1, dst1)  #.wav
        #chuyen mp3->wav 16000Hz
        src2=dst1
        dst2=dst1.replace("mp3", "wav")
        convertMp3ToWav16(src2, dst2)
        os.remove(dst1)
        path = dst2
    if (file_extension.lower()==".mp3"):
            #chuyen mp3->wav 16000Hz
        src= temp_audio_path
        dst=src.replace("mp3", "wav")
        convertMp3ToWav16(src, dst)
        path = dst
    if (file_extension.lower()!=".wav"):
        os.remove(temp_audio_path)

    return path