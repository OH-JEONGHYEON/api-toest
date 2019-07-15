# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:24:59 2019

@author: JH
"""

import asyncio
import subprocess
from ibm_watson import SpeechToTextV1

from time import time
from os.path import join, dirname

class STT:

    def __init__(self, path=None):

        self.loop = asyncio.get_event_loop()
        # script = []
        self.sourcefile = path


    def run(self, path=None):

        src = path or self.sourcefile
        self.convertfile = self.convert2flac(src)
        self.block_num, self.mutes = self.find_mute()
        self.request_stt() # -> self.result
        for re in self.result:
            for j, sen in enumerate(re['results']):
                print(j, self.ans_index(sen['alternatives'][0]['transcript']))
            print()

        return self.result


    def convert2flac(self, path=None):

        src = path or self.sourcefile
        real_file = src.split('/')[-1].split('.')[0]
        dest = "./tmp/{}.flac".format(real_file)
        comm = "ffmpeg -i {} -y -loglevel panic -ac 1 -sample_fmt s16 -ar 48000 {}".format(src, dest)
        subprocess.call(comm, shell=True)

        return dest


    def find_mute(self, path=None):

        src = path or self.convertfile

        comm = "ffmpeg -hide_banner -i {} -af silencedetect=n=-50dB:d=1 -f null -".format(src)

        output = subprocess.check_output(comm,
                                         shell=True,
                                         stderr=subprocess.STDOUT,
                                         universal_newlines=True)

        res = output.split('\n')
        imp = [r for r in res if 'silencedetect @' in r]

        rng = len(imp) // 2 - 1
        if float(imp[0].split(' ')[4]) == 0:
            imp.pop(0)
        else:
            time_stamp = [[0.0, float(imp[0].split(' ')[4])]]
        for i in range(rng):
            time_stamp.append([float(imp[2 * i + 1].split(' ')[4]), float(imp[2 * i + 2].split(' ')[4])])

        block_num = len(time_stamp)
        print("block number: ", len(time_stamp), end='\n\n')

        return block_num, time_stamp


    def request_stt(self):

        self.loop.run_until_complete(self._main())
        self.loop.close

        # return self.result

    def ans_index(self, sen):
        #     s = re.sub('[^A-Za-z0-9]+', '', sen).lower().strip()
        s = sen.replace('.', ' ').lower().strip().split(' ')
        chk = s[0]
        if 'hey' == chk:
            s[0] = '(A)'
        if 'be' == chk or 'b' == chk:
            s[0] = '(B)'
        if 'see' == chk or 'c' == chk:
            s[0] = '(C)'
        if 'd' == chk:
            s[0] = '(D)'
        return " ".join(s)


    async def _main(self):

        fts = [asyncio.ensure_future(self._main2(idx, m)) for idx, m in enumerate(self.mutes)]
        self.result = await asyncio.gather(*fts)


    async def _main2(self, idx, mute):

        # path = await self.loop.run_in_executor(None, self.chunk_audio, *[idx, mute])
        path = await self.chunk_audio(idx, mute)
        # res = await self.loop.run_in_executor(None, self.stt_audio, path)
        res = await self.stt_audio(path)

        return res


    async def chunk_audio(self, idx, mute):

        path = "./tmp/{}.flac".format(idx)
        comm = ['ffmpeg', '-i', self.convertfile, '-y', '-loglevel', 'panic', '-ss', str(mute[0]), '-to', str(mute[1]), path]

        print("start chunk...", idx, end='\n\n')
        start = time()
        await self.loop.run_in_executor(None, subprocess.call, comm)
        end = time()
        print("end chunk...", idx, end-start, end='\n\n')

        return path


    async def stt_audio(self, path):

        print("start stt.....", path, end='\n\n')

        with open(path, 'rb') as audio_file:
            speech_recognition_results = await self.loop.run_in_executor(None, self._wrapper_ibm_stt, audio_file)

            # print(speech_recognition_results, '\n\n')

            print("done.", path)

            return speech_recognition_results


    def _wrapper_ibm_stt(self, audio):

        speech_to_text = SpeechToTextV1(
            iam_apikey="iB_ldfwxsW5QLIfKR_G629EzkiZrb-gr1kDo45tbE2fb",
            url="https://gateway-tok.watsonplatform.net/speech-to-text/api"
        )

        speech_recognition_results = speech_to_text.recognize(
            audio=audio,
            content_type='audio/flac',
            word_alternatives_threshold=0.9
        ).get_result()

        return speech_recognition_results



if __name__=="__main__":

    start = time()

    subprocess.call('rm -rf /tmp/*.mp3', shell=True)  # linux
    subprocess.call('rm -rf /tmp/*.flac', shell=True)  # linux

    path = './audio/test1.mp3'

    stt = STT(path)

    stt.run()

    end = time()

    print("total run time: ", end-start, end='\n\n')