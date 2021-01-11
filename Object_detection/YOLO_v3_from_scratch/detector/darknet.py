# 필요한 library import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# parse_cfg 함수 정의하기, 구성 파일의 경로를 입력으로 받습니다.
def parse_cfg(cfgfile):
    '''
    configuration 파일을 입력으로 받습니다.

    blocks의 list를 반환합니다. 각 blocks는 신경망에서 구축되어지는 block을 의미합니다.
    block는 list안에 dictionary로 나타냅니다.
    '''

    # cfg 파일 전처리
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')               # lines를 list로 저장합니다.
    lines = [x for x in lines if len(x) > 0]      # 빈 lines를 삭제합니다.
    lines = [x for x in lines if x[0] != '#']     # 주석을 삭제합니다.
    lines = [x.rstrip().lstrip() for x in lines]  # 공백을 제거합니다.

    # blocks를 얻기 위해 결과 list를 반복합니다.
    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':              # 새로운 block의 시작을 표시합니다.
            if len(block) != 0:         # block이 비어있지 않으면, 이전 block의 값을 저장합니다.
                blocks.append(block)    # 이것을 blocks list에 추가합니다.
                block = {}              # block을 초기화 합니다.
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks
