# @Author: Alfonsas Jursenas
# @Date:   2020-04-18T19:33:08+03:00
# @Email:  alfonsas.jursenas@gmail.com
# @Last modified by:   Alfonsas Jursenas
# @Last modified time: 2020-04-18T19:51:35+03:00


import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
from collections import defaultdict
from numpy import floor
import gc


from . import utils
from .gen_utils import superimpose_img, select_char_example, get_imgs_dict
from .gen_utils import rotateImage
from skimage.color import label2rgb

import scipy




# char_img = get_imgs_dict('input/numbers/')
# char_img_test = get_imgs_dict('input/numbers_test/')
bg_paths_train = glob('input/backgrounds/*.jpg')
bg_paths_test = glob('input/backgrounds_test/*.jpg')

# text_to_idx ={
#     '0':1, '1':2, '2':3, '3':4, '4':5, '5':6, '6':7, '7':8, '8':9, '9':10,
#     '.': 11, '-': 12
#     }
# idx_to_text = dict((v,k) for k,v in text_to_idx.items())


# =============================== Misc functions ===============================


def pad_and_rotate(img, angle, padding = 16, pad_value=(255,255,255,0)):
    '''
    Multidimensional image rotate and pad

    img --- rgba float [0,255] img
    angle --- angle in degrees
    TLBR -- padding px
    '''
    img2 = np.copy(np.float32(img))

    slices = []

    if len(img2.shape)==2:
        img2 = cv2.copyMakeBorder(img2, top=padding, bottom=padding, left=padding, right=padding, borderType=cv2.BORDER_CONSTANT, value=0 )
        img2 = rotateImage(img2, angle, borderValue = 0)

    else:
        # go over all dims
        for f in range(img2.shape[2]):
            slices.append( cv2.copyMakeBorder(img2[:,:,f], top=padding, bottom=padding, left=padding, right=padding, borderType=cv2.BORDER_CONSTANT, value=0 ))
            slices[-1] = rotateImage(slices[-1], angle, borderValue = 0)

        img2 = np.stack(slices, axis=2)


#     img2 = scipy.ndimage.rotate(img2, angle, cval=0.0)

    return img2



def norm(x): return x / x.max()



def get2D_classes_mask(img2, classes_mask):
    '''
    Convert to 2D classes mask
    '''
    n_classes, _ = classes_mask.shape
    h, w, _, = img2.shape

    # expand mask
    classes_mask2d = np.zeros((h, w, n_classes))
    classes_mask2d[:,:,:] = np.expand_dims(classes_mask.T, axis=0)
    classes_mask2d_debug = np.argmax(classes_mask2d, axis=2)

    return classes_mask2d


def smooth(x, window_len=20):
    """smooth the data using a window with requested size
    """

    w = np.hanning(window_len)

    y = np.convolve(w/w.sum(), x, mode='same')
    return y


def char_class_mask_norm(mask):
    '''normalize over classes'''
    norm = np.sum(mask, 0, keepdims=True)
    norm[norm==0] = 1
    return mask / norm


def eps_smooth(mask, eps=0.05):
    '''
    class labels epsilon smoothing

    !mutates mask!
    '''
    mask = np.copy(mask)

    mask[mask==0] += eps
    mask[mask==1] -= eps
    return char_class_mask_norm(mask)

def smooth2D_X(mask):

    mask = np.copy(mask)
    nx, ny = mask.shape
    for n in range(nx):
        mask[n,:]=smooth(mask[n,:])

    # probability normalisation
    return char_class_mask_norm(mask)


# ================================= Plan image =================================

def plan_image(
    char_width = 32,
    D_Width = 5, #
    char_height = 64,
    D_Height = 5, #
    spacing = 32,
    D_Spacing = 5,
    text = "XX.XX.XX   X"
):
    '''
    Generate image plans


    return
    gt_text --- gt text
    widths --- width of each cahr
    heights --- height of each char
    spaces --- space after each char
    top_margin --- top margin for each char
    '''


    # --------------- allocate lists ---------------
    gt_text = [] # GT text
    widths = []
    heights = []
    spaces = []
    top_margin = []

    # random number [-1,1]
    ints = lambda X : [int(x) for x in X]
    rnd = lambda : (2*np.random.rand() - 1)


    # generating "plans" for the image
    for char_type in text:

        # multiplicative distortions
        dw = 1 + D_Width / 100 * rnd()
        dh = 1 + D_Height / 100 * rnd()
        ds = 1 + D_Spacing / 100 * rnd()


        # Character cases
        if char_type == 'X':
            char = str(np.random.randint(0, 9 + 1))
            gt_text.append(char)
            top_margin.append(0)
            spaces.append(spacing * ds) # space after
            widths.append(char_width * dw )
            heights.append(char_height * dh )

        if char_type == ' ':
            gt_text.append(' ')
            top_margin.append(0)
            spaces.append(0) # space after
            widths.append(char_width * dw )
            heights.append(0)


        if char_type == '.':
            char = '.'
            gt_text.append(char)
            top_margin.append(64 - char_height // 3 * dh)
            spaces[-1] = spacing // 2 * ds # spacing before
            spaces.append(spacing // 2 * ds) # spacing after

            # decide on character width, height
            widths.append(char_width // 3 * dw )
            heights.append(char_height // 3 * dh )

    return gt_text, ints(widths), ints(heights), ints(spaces), ints(top_margin)


# ========================= Generate characters image ==========================
def characters_rgba(char_img,
                    text_to_idx,
                    params = False,
                    text = "XX.XX.XX   X"
                    ):
    '''
    Generate empty image with characters + character classes image

    img_size --- output RGBA image size
    char_img --- dict of character images
    char_width --- single character image width
    D_Width          --- max width distortion +-%
    char_height --- single character image height
    D_Height          --- max height distortion +-%
    spacing --- spacing between characters
    D_spacing --- max spacing distortion +-%


    deformations happen by sampling [-1,1] random nr
    '''
    if not params:
        params = {
            'char_width' : 32,
            'D_Width' : 10,
            'char_height' : 64,
            'D_Height' : 5,
            'spacing' : 16,
            'D_Spacing' : 20,
            'text' : "XX.XX.XX   X"
        }




    gt_text, widths, heights, spaces, top_margin  = plan_image(**params)




    # Generate background image to contain all chars exactly
    # use white transperant background

    # numbers image size
    height = np.max(
                (
                    np.max(heights), # char heights
                    np.min(heights) + np.max(top_margin) # small chars with margin
                )
            )
    width = np.sum(widths)+np.sum(spaces[0:-1]) # ignore last space


    classes_mask = np.zeros((len(text_to_idx)+1, width)) # a mask for CE loss, +1 due to blank

    # here should be the original background loaded?
    # Transperent background
    background = np.ones((height, width, 4), dtype=np.float32)*255
    background[:,:,3] = 0.0

    x_coord = 0

    gt_text2 = [] # gt text to be used

    # Superimpose character images
    for n in range(len(gt_text)):


        char = gt_text[n]

        if char == ' ':
            # WARNING -- BLANK=0 ASUMPTION !!!!
            classes_mask[0, x_coord:x_coord+widths[n]] = 1.0

            x_coord += widths[n]
            continue

        gt_text2.append(char) # append all chars except ' '

        # select the random character example
        example_idx = select_char_example(char_img, char)


        # CE label
        x_center = x_coord + widths[n] // 2

        # reducing character witdth in class mask
        dw = int(widths[n] / 3)
        if char == '.':
            dw = int(widths[n] / 1.5)

        classes_mask[text_to_idx[char], x_center - dw : x_center + dw] = 1.0
        # balnk label in spacing after
        classes_mask[0, x_coord+widths[n]:x_coord+widths[n]+spaces[n]] = 1.0

        # character onto image
        char_img_ = char_img[char][example_idx]

        if top_margin[n] > 0:
             top_margin[n] -= 1

             
        img2 = superimpose_img(
                            background,
                            char_img_,
                            (0, x_coord), # y_coord, x_coord
                            char_size = (widths[n], heights[n]),
                            top_margin = top_margin[n]
        )
        del char_img_
        gc.collect()

        x_coord += spaces[n] + widths[n] # take a step

    del background
    gc.collect()


#     img2 = cv2.cvtColor(np.uint8(img2), cv2.COLOR_RGBA2RGB)
    return img2, gt_text2, classes_mask




# =========================== Cut random Background ============================

def random_bg_cut(bg, height = 128, width = 576):
    # ----- cut to needed shape + random translation ------
    bg1 = np.copy(bg)
    height0, width0, _ = bg1.shape

    Dx = width0 - width
    Dy = height0 - height

    # random translation


    # if image is too small
    if (Dx < 0) and  (Dy < 0):
        dx=0
        dy=0
        bg1 = cv2.resize(bg, (width, height))

    elif(Dx < 0):

        bg1 = cv2.resize(bg, (width, height0))
        dx=0
        dy = np.random.randint(0, Dy)

    elif(Dy < 0):

        bg1 = cv2.resize(bg, (width0, height))
        dx = np.random.randint(0, Dx)
        dy = 0
    else:
        # image is larger than needed
        dx = np.random.randint(0, Dx)
        dy = np.random.randint(0, Dy)

    bg1 = bg1[dy:height+dy, dx:width+dx]

    return bg1


# ============================= Random stamp date =============================
# ==============================================================================

def random_stamp_date2(char_img, bg_paths_train, text_to_idx, max_angle=5):

    '''
    Returns
    img3, gt_text2, classes_mask2_small, classes_mask_proj_small, classes_mask2 = random_stamp_date(char_img)
    '''

    # ========================== Make the background ===========================
    bg = cv2.imread(np.random.choice(bg_paths_train)) # random bg
    bg = cv2.cvtColor(np.uint8(bg), cv2.COLOR_RGBA2RGB)
    bg = random_bg_cut(bg, height = 128, width = 576)
    # add alpha dimension to the bg
    bg_ = np.ones((128,576,4))*255
    bg_[:,:,0:3] = bg

    # ======================= Transperant numbers image ========================
    # generate the numbers
    img2, gt_text2, classes_mask = characters_rgba(char_img, text_to_idx)

    # blur numbers alpha channel
    # img2[:,:,3]  = cv2.GaussianBlur(np.uint8(img2[:,:,3]), (3,3), 0)

    # ========================== get 2d classes mask ===========================
    # converts  (height, width, class) ---> (height, width) image of class integers
    classes_mask_2d = get2D_classes_mask(img2, classes_mask)
    #classes_mask_2d_labels = np.argmax(classes_mask_2d, axis=2)


    # =============== pad and rotate: characters and class-mask ================
    # image rotation
    angle = np.random.randint(-max_angle,max_angle) # select rotation angle [deg]
    img2 = pad_and_rotate(img2, angle)

    # class mask rotation
    # WARNING: will work only with small rotations
    # todo: could also work with down-sampled image to optimize ?
    classes_mask_rot = pad_and_rotate(classes_mask_2d, angle=angle, pad_value=0)
    classes_mask_rot = np.argmax(classes_mask_rot, 2) # max over class dimension

    # ====== superimpose numbers onto background,  classes onto empty bg =======
    h0, w0, _ = bg_.shape
    h1, w1, _ = img2.shape

    # random translation -- same for classes and image
    dx = np.random.randint(0, w0 - w1)
    dy = np.random.randint(0, h0 - h1)

    img3 = superimpose_img(bg_, img2, topleft=(dy,dx), char_size=(img2.shape[1],img2.shape[0]))
    img3 = np.asarray(img3 / img3.max() * 255, dtype=np.uint8)

    classes_mask2 = np.zeros((h0, w0)) #bg_[:,:,0]*0
    classes_mask2[dy:h1+dy, dx:w1+dx] = classes_mask_rot

    # ================== Small classes mask and x-projection ===================
    classes_mask2_small = classes_mask2[:, ::16] # stride 16 downsampling x-axis
    # contracting y-axis to get x-projection
    classes_mask_proj_small = np.max(classes_mask2_small, axis=0)

    img3 = cv2.cvtColor(img3, cv2.COLOR_RGBA2RGB) / 255.0

    return  img3, gt_text2, classes_mask2_small, classes_mask_proj_small, classes_mask2
