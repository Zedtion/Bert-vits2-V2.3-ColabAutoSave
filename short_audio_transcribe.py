# -*- coding:utf-8 -*-

import traceback
import os
import sys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

dataset_name = 'esd'
read_dir = './Data/ada/raw'
write_dir = './Data/ada/wavs'
# opt_name=dir.split("\\")[-1].split("/")[-1]
# opt_name = os.path.basename(dir)
opt_name = 'ada'
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='./damo_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    vad_model='./damo_models/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    punc_model='./damo_models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
)

opt = []
for name in os.listdir(read_dir):
    print(name)
    try:
        text = inference_pipeline(input="%s/%s" % (read_dir, name))[0]['text']
        opt.append("%s/%s|%s|ZH|%s" % (write_dir, name, opt_name, text))
    except:
        print(traceback.format_exc())

opt_dir = "./Data/ada"
os.makedirs(opt_dir, exist_ok=True)
with open(f"{opt_dir}/{dataset_name}.list", "w", encoding="utf-8")as f:
    f.write("\n".join(opt))
