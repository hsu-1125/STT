from __future__ import absolute_import, division, print_function
​
import os
import numpy as np
import shlex
import subprocess
import sys
import wave
import requests
​
brew = os.system('which -s brew')
if brew == 0 : 
    print('brew installed.')
else :
    print('brew not installed.')
    os.system("ruby -e $(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install)")
​
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib.request
urllib.request.urlretrieve('https://itutbox.s3.amazonaws.com/voice/production/BD2FB1FA-AAA4-4D03-94E0-E178C3278A46.mp4', 'speech.mp4') 
​
# deepspeech module
dp = os.system('pip show deepspeech')
if dp == 0 : 
    print('deepspeech installed.')
else :
    print('deepspeech not installed.')
    os.system("pip install deepspeech")
​
import deepspeech
from deepspeech import Model, version
from timeit import default_timer as timer
​
# """# Accoustic parameters"""
def checkDeepSpeechFile(str, mode):
    isModelExists = os.path.isfile(str)
​
    if isModelExists :
        print('File exist')
    else :    
        print('File not exist')
     
        if mode == 0:
            urllib.request.urlretrieve('https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm')
        else :
            urllib.request.urlretrieve('https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer')
​
model  = '/home/ubuntu/STT/deepspeech-0.9.3-models.pbmm'
scorer = '/home/ubuntu/STT/deepspeech-0.9.3-models.scorer'
​
checkDeepSpeechFile(model,0)
checkDeepSpeechFile(scorer,1)
​
# These constants control the beam search decoder
​
# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 100
​
# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 0.76
​
# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.8
​
# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training
​
# Number of MFCC features to use
# N_FEATURES = 32
​
# Size of the context window used for producing timesteps in the input vector
# N_CONTEXT = 12
​
"""# Adapt Sample Rate of Audio File"""
def convert_samplerate(audio_path):
    # subprocess.call(['ffmpeg','-y', '-i', 'speech.wav', '-ac', '1', '-ar', '16000' , 'speech.wav'])
    os.system('ffmpeg -y -i speech.mp4 -ar 16000 -ac 1 -f wav speech.wav')
    return 16000, np.frombuffer('speech.wav', np.int16)
​
"""# Input MP4 Audio File"""
# convert to wav file.  
package = os.system('brew ls --versions ffmpeg')
if package == 0 :
    print(' The package is installed')
else :
    print('The package is not installed')
    os.system('brew install ffmpeg')
​
# subprocess.call(['ffmpeg','-y', '-i', 'speech.mp4', '-ac', '1', '-ar', '16000', '-f', 'wav' , 'speech.wav'])
os.system('ffmpeg -y -i speech.mp4 -ar 16000 -ac 1 -f wav speech.wav')
​
"""# Convert MP3 to Text"""
audio = 'speech.wav'   
    
print('Loading model from file {}'.format(model), file=sys.stderr)
model_load_start = timer()
ds = Model(model)
model_load_end = timer() - model_load_start
print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)
lm_load_start = timer()
ds.setBeamWidth(BEAM_WIDTH)
ds.enableExternalScorer(scorer)
ds.setScorerAlphaBeta(LM_WEIGHT,VALID_WORD_COUNT_WEIGHT)
lm_load_end = timer() - lm_load_start
    
fin = wave.open(audio, 'rb')
fs = fin.getframerate()
if fs != 16000:
    print('Warning: original sample rate ({}) is different than 16kHz. Resampling might produce erratic speech recognition.'.format(fs), file=sys.stderr)
    fs, audio = convert_samplerate(audio)
else:
    bf = fin.readframes(fin.getnframes())
    audio = np.frombuffer(bf, np.int16)
audio_length = fin.getnframes() * (1/16000)
fin.close()
print('Running inference.', file=sys.stderr)
print('================================\n')
inference_start = timer()
print(ds.stt(audio))
inference_end = timer() - inference_start
print('\n================================')
print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)

postdata={"uuid":str(ds.stt(audio))}
url = "https://api.italkutalk.com/api/Apple/Signature"
requests.post(url,json=postdata,)