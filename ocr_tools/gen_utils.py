import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
from collections import defaultdict
from numpy import floor
import gc


def get_imgs_dict(path='img/numbers/',
                  chars = ['-','.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]):

    '''
    Return a dictionary containing lists of symbol image arrays
    '''

    char_img_paths = {}

    for c in chars:
        char_img_paths[c] = glob(path + c +'*')


    char_img = defaultdict(list)

    for char, paths in char_img_paths.items():
        for path in paths:
            char_img[char].append(
                cv2.cvtColor(
                    cv2.imread(path, cv2.IMREAD_UNCHANGED),
                cv2.COLOR_BGRA2RGBA
                )
            )
    return char_img



# ============================= Superimpose chars ==============================
from skimage.morphology import remove_small_objects
from .poisson import process as poisson_blend

def superimpose_img(background,
                    img,
                    topleft,
                    char_size=(32, 64),
                    poisson = False,
                    top_margin = 0 ):
    '''
    Put character onto the background

    background --- 4 chnl background
    img        --- the imge to be superimposed
    '''
    img = cv2.resize(img, char_size, interpolation=cv2.INTER_CUBIC)

    # use alpha channel to kill background
    alpha = np.expand_dims(img[:,:,3], axis=2)
    alpha = alpha / np.max(alpha)

    if not poisson:
        img = img * alpha


    top, left = topleft
    top = top + top_margin
    height, width, _ = img.shape


    if poisson:
        mask_bw = remove_small_objects(alpha > 0.2, min_size=4)
        result_stack = [poisson_blend(
                            img[:,:,i],
                            background[top:top + height, left:left+width, i],
                            mask_bw) for i in range(3)]

        background = cv2.merge(result_stack)
        del mask_bw
    else:
        # softly remove the background where the symbol must be
        background[top:top + height, left:left+width] *= (1-alpha)
        # superimpose the image onto the background
        background[top:top + height, left:left+width] += img

    # Mem clean
    del img
    del alpha

    return np.asarray(background, dtype = np.int32)


# ------------------------------------------------------------------------------
# ---------------------------- Random stamp numbers ----------------------------
# ------------------------------------------------------------------------------

def random_stamp_numbers(char_img):
    '''
    Generate an image of random numbers drawn from img/numbers folder
    '''
    # TODO: make global
    n_characters = 8
    char_width = 32
    char_height = 64
    spacing = 32

    # use simple white background
    background = np.ones((char_height, n_characters*char_width, 4))*255

    text = [] # GT text

    for n in range(8):
        number = np.random.randint(0, 9+1)
        char = str(number)
        text.append(char)

        # select the character example
        n_examples = len(char_img[char])
        if n_examples > 1:
            example_idx = np.random.randint(0, n_examples-1 )
        else:
            example_idx = 0


        img2 = superimpose_img((background), np.copy(char_img[char][example_idx]), (0,spacing*n))

        # rgba to rgb
        img2 = cv2.cvtColor(np.uint8(img2), cv2.COLOR_RGBA2RGB)


    return img2 / 255.0, text

# plt.imshow(img2)
# plt.show()

# img, text = random_stamp_numbers(char_img_test)
# plt.imshow(img)
# print(text)


# =========================== Small helper functions ===========================
# --------------------------------------  --------------------------------------
# -------------------------------- Rotate image --------------------------------
# ------------------------------------------------------------------------------
def rotateImage(image, angle, borderValue = (255, 255, 255) ):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_LINEAR, # may need to change to cubic
                            borderValue=borderValue
                         )
    return result


# ---------------------- Select a character image version ----------------------
def select_char_example(char_img, char):
    '''
    Select a random character from a list of its versions
    char_img --- dict of character images
    '''

    n_examples = len(char_img[char])
    assert n_examples > 0

    if n_examples > 1:
        example_idx = np.random.randint(0, n_examples-1 )
    else:
        example_idx = 0

    return  example_idx

# ==============================================================================
# =========================== RANDOM DATE GENERATION ===========================
# ==============================================================================
import gc

def random_stamp_date(char_img):
    '''
    Generate an image of random numbers drawn from img/numbers folder
    '''
    # TODO: make global
    #n_characters = 8
    char_width = 32
    char_height = 64
    spacing = 32

    # use simple white background
    background = np.ones((char_height, 512, 4))*255

    text = [] # GT text

    # Date format XX.XX.XX    Z

    x_coord = 0
    for char_type in "XX.XX.XX   X":


        # Character cases
        if char_type == 'X':
            number = np.random.randint(0, 9 + 1)
            char = str(number)
            char_size_ = (32,64) # randomize?
            text.append(char)
            spacing = 32
            top_margin = 0

        if char_type == '.':
            char = '.'
            char_size_ = (16,16) # randomize?
            text.append(char)
            spacing = 16
            top_margin = 64-16

        # for space just skip
        if char_type == " ":
            x_coord += spacing_ # step horizontally
            continue


        # randomize spacing
        spacing_ = int(spacing* (1+np.random.rand()*0.3) )


        # select the random character example
        example_idx = select_char_example(char_img, char)


        # character onto image


        if top_margin > 0: top_margin -=1 # minor correction

        img2 = superimpose_img(
                            background,
                            np.copy( char_img[char][example_idx] ),
                            (0, x_coord),
                            char_size = char_size_,
                            top_margin = top_margin
        )



        # step horizontally
        x_coord += spacing_


        # rgba to rgb
        img2 = cv2.cvtColor(np.uint8(img2), cv2.COLOR_RGBA2RGB)


    # Add whitespace margin and rotate sligltly
    #ia2 = (np.ones(img2.shape[0]+64, img2.shape[1]+64, 3)*255)[]
    img2 = cv2.copyMakeBorder(img2, top=32, bottom=32, left=64, right=0, borderType=cv2.BORDER_CONSTANT, value=(255,255,255) )

    img2 = rotateImage(img2, np.random.randint(-5,5) )



    # downsample resolution for half of examples
    if np.random.rand() > 0.5:

        if np.random.rand() > 0.5:
            img2 = cv2.resize(img2, (137,48)) # small size
        else:
            img2 = cv2.resize(img2, (100,20)) # small size

        img2 = cv2.resize(img2, (576,128)) #, the size needed for network



    # memory cleanup
    del background
    gc.collect() # TODO: may be not needed in the future


    return img2/255.0, text
