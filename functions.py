import numpy as np
from progress_timer import Timer

def enc_dec_split(data, special_tok):
    enc, dec = [], []
    timer = Timer(len(data))
    for idx, line in enumerate(data):
        timer.time_check(idx)
        enc.append(line[:5])
        dec.append([special_tok] + line[5:])

        if idx == 0:
            print("original:", line)
            print('enc:', enc[0])
            print('dec:', dec[0])
    return np.array(enc), np.array(dec)
