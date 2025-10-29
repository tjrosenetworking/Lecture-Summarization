#pre-requisites 
#pip install transformers
#pip install torch
#pip install hf_xet
#python 3.13 

import torch
import transformers

print(f' GPU?: {torch.cuda.is_available()}') 
#torch only supports nvidia gpus #:madface: and the AMD methods aren't as well supported
#gpu would help with processing times (if available on your pc)

from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0)


#actual transcription
print("BEGINNING TRANSCRIPTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
transcription = pipe(r"[INSERT YOUR FILE PATH]", return_timestamps=True, language='en')

#processing for the 13 minute video took 4:06 on my laptop 
print("FINISHED TRANSCRIPTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# print(transcription) this output has timestamps which we do not want 

cleanOutput = [] 
for entry in transcription: #this code removes timestamps and makes it only a textual output 
    if 'text' in transcription:
        cleanOutput.append(transcription['text'])

print(cleanOutput)