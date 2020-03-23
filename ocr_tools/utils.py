import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import defaultdict
from numpy import floor
import torch
# ==============================================================================
# ====================== Estimate tensor shape by layers =======================
# ==============================================================================
# # %% codecell
# def conv2d(height, width, kernel, stride=(1,1)):
#     return  int(floor((height - kernel)/stride[0] + 1))  , \
#             int(floor((width - kernel )/stride[1]) + 1)
#
# def maxpool2d(height, width, kernel, stride = 1):
#     return (height//kernel)
# # %% codecell
# height, width = (64, 256)
# kernels = [3,5,14,16]
# strides = [(1,1),(1,1),(2,2),(2,2)]
#
# for n in range(len(kernels)):
#
#     k = kernels[n]
#     stride = strides[n]
#
#     height, width =conv2d(height,width, k, stride)
#     print(height, width)

# ==============================================================================



# ==============================================================================
# ==========================CHARACTER(WORD) ERROR RATE==========================
# ==============================================================================
def wer(r, h):
    '''
    Word error count computation

    https://en.wikipedia.org/wiki/Word_error_rate ?

    sequences of integers []
    r --- prediction
    h --- ground truth
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1)) # why not create coorect dims at start?
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


# ==============================================================================
# ======================= Convert predictions to integer =======================
# ==============================================================================
def preds_to_integer(Preds, eps, p_tresh = 0.5):
    '''`
    Preds --- (log?) probabilities [Character class, Horizontal dim]
                        [n_batches, n_character_classes, width]
    Batch example is selected!
    p_tresh --- treshold for character detection
    eps --- Blank char index

    Returns
        out --- predicted char indexes
        p_out --- predicted char probs

    '''
    preds=torch.argmax(Preds, dim=0).detach().cpu().numpy()

    # take maximally likely characters
    probs=np.exp(
            np.max(Preds.detach().cpu().numpy(), axis=0)# max over probability
    )

    preds=preds.tolist()
    out = [] # prediction char indexes
    p_out = [] # prob of that char

    # collapse repeating characters and epsilon
    for i in range(len(preds)):
        '''
        if char is not eps
        anf if char is not the same as previous one
        '''
        if preds[i] != eps and preds[i] != preds[i - 1]:
            if probs[i] > p_tresh: # if the character is likely enough
                out.append(preds[i])
                p_out.append(probs[i])

    return out, p_out


# ------------------------------------------------------------------------------
# ------------------- Convert preds to integer and eval WER --------------------
# ------------------------------------------------------------------------------
# TODO: May need to change eps if adding more characters

def wer_eval(preds, labels, eps = 12, ratio = False):
    '''
    Word (character) error rate evaluation

    preds  --- log probabilities [T, C]
    labels --- int labels (?)
    eps --- blank char index,
        = len(text_to_idx)

    1. Convert predictions to integers
    2. Eval WER accuracy
    '''

    # Collapse blanks and take maximaly likely characters
    preds, probs = preds_to_integer(preds, eps, p_tresh=0.5)

#     print('preds:', preds)
#     print('labels:', labels)
    we = wer(preds, labels)

    # just return nuymber of errrors
    if ratio == False:
        return we


    N = len(preds)

    # nothing to predict and nothing is predicted
    if len(labels) == 0 and N==0: return 0

    if N==0:
        # if nothing is predicted, than we made 100% errors on each true character
        return 1

    return we/len(preds)
