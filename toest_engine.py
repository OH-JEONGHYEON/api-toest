import asyncio
import os
import string
import subprocess
import random

from deepspeech import Model
import wave
import numpy as np

from time import time


class STT():

    def __init__(self, src):

        self.src = src
        self.SAMPLE_RATE = 16000 # 나중에 바꿔볼까...?

        # STT 모델 로드
        model = './deepspeech-0.5.1-models/output_graph.pbmm'
        alphabet = './deepspeech-0.5.1-models/alphabet.txt'
        lm = './deepspeech-0.5.1-models/lm.binary'
        trie = './deepspeech-0.5.1-models/trie'
        self.ds = Model(model, 26, 9, alphabet, 500)
        self.ds.enableDecoderWithLM(alphabet, lm, trie, 0.75, 1.85)

        self.loop = asyncio.get_event_loop()

    def run(self):

        self.folder, self.cnvt = self.convert2wav()

        self.num, self.time_stamp = self.find_mute()

        self.loop.run_until_complete(self.gather_futures())
        self.loop.close()
        # 결과 self.result에 저장

        subprocess.call('rm -rf ./tmp/{}'.format(self.folder), shell=True) # wav음원 삭제

        return self.result


    # 음원파일(mp3) -> wav 파일로 변환
    def convert2wav(self):

        filename = self.src.split('/')[-1].split('.')[0] #진짜 파일이름

        # 임의 폴더 생성
        folder = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
        os.mkdir('./tmp/'+folder)

        dest = "./tmp/{}/{}.wav".format(folder,filename) # 폴더명 겹칠 경우를 생각해야하나?

        try:
            comm = "ffmpeg -y -i {} -ac 1 -acodec pcm_s16le -ar 16000 {}".format(self.src, dest)
            subprocess.check_output(comm,
                                    shell=True,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
        except Exception as e:
            print("==convert ERROR!==")
            print(e)

        return folder, dest

    # 무음구간 디텍트
    def find_mute(self):

        try:
            comm = "ffmpeg -hide_banner -i {} -af silencedetect=n=-80dB:d=0.3 -f null -".format(self.cnvt)
            output = subprocess.check_output(comm,
                                             shell=True,
                                             stderr=subprocess.STDOUT, # 이거 차이를 잘 모르것네
                                             universal_newlines=True)
        except Exception as e:
            print("==fine mute ERROR!==")
            print(e)

        res = output.split('\n')
        imp = [r for r in res if 'silencedetect @' in r]

        # for r in imp:
        #     print(r)
        # print("imp: ",len(imp))

        if float(imp[0].split(' ')[4]) == 0:
            time_stamp = []
        else:
            time_stamp = [[0.0, float(imp[0].split(' ')[4])]]

        for i in range(1, len(imp)-1, 2):
            print(i)
            sen_start = float(imp[i].split(' ')[4])
            sen_end = float(imp[i+1].split(' ')[4])
            time_stamp.append((sen_start, sen_end))

        num = len(time_stamp)
        # for t in time_stamp:
        #     print(t)
        # print("num: ",num)

        return num, time_stamp

    async def gather_futures(self):

        futures = [asyncio.ensure_future(self.main(idx, time)) for idx, time in enumerate(self.time_stamp)]
        self.result = await asyncio.gather(*futures)

    async def main(self, idx, time):

        print("start chunk audio... "+str(idx), end='\n\n')
        dest = './tmp/{}/{}.wav'.format(self.folder, idx)
        comm = ['ffmpeg', '-i', self.cnvt, '-y', '-loglevel', 'panic',
                '-ss', str(time[0]), '-to', str(time[1]), dest]

        await self.loop.run_in_executor(None, subprocess.call, comm)

        print("start stt audio... "+str(idx), end='\n\n')
        fin = wave.open(dest, 'rb')
        fs = fin.getframerate()

        if fs != self.SAMPLE_RATE:
            print('not same sample rate')
            exit(1)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

        audio_length = fin.getnframes() * (1 / self.SAMPLE_RATE)
        fin.close()

        return self.ds.stt(audio, fs)


if __name__=='__main__':

    source = './audio/test01.mp3' # MP3 음원 파일

    stt_start = time()
    stt = STT(source)
    result = stt.run()
    stt_end = time() - stt_start


    for i, r in enumerate(result):
        print(i, r)

    print()
    print("total time(s): ", stt_end)

