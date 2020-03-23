import torch
import numpy as np

class IterDataset(torch.utils.data.IterableDataset):
    '''
    Return example pairs
        * x, y
            * x --- image
            * y --- a list of character indexes
        * No need to form a batch at this point
            * Data loader does this
        * Provide the size of the dataset


    Overwrite __iter__(), which would return an iterator of samples in this dataset

    the samples need to be in batches ?


    When num_workers > 0, each worker process will have a different copy of the
    dataset object, so it is often desired to configure each copy independently to
    avoid having duplicate data returned from the workers.
    '''

    def __init__(self, char_images_dict, random_stamp_date, examples_count = 64*100):
        '''
        char_images_dict --- mapping of characters to images
        random_stamp_date --- function generating random date
        examples_count   --- the number of examples in the dataset
        '''
        super(IterDataset).__init__()

        # init attributes
        self.char_images_dict = char_images_dict.copy() # copy for safety, this is small dictionary.
        self.examples_count = examples_count
        self.random_stamp_date = random_stamp_date


    def generator(self):
        '''
        Generates a single example in non-tensor form

        char_img       ---  images of characters
                            these images should be separate for train/test sets
        examples_count ---  number of examples in the

        Returns: img, y_gt
            img  --- rgb image [np array]
            y_gt --- ground truth indexes of the single example [list]
        '''

        for _ in range(self.examples_count):

            # stitching together the char-images to create an example
            img, text = self.random_stamp_date(self.char_images_dict)

            # integers representation of text
            y_gt = [text_to_idx[t] for t in text]

            yield torch.tensor(img, dtype=torch.float), torch.tensor(y_gt)


    def __iter__(self):
        '''Return an iterator to the dataset'''
        # Logic for parallel vs single process implementation
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pass
            # single-process data loading, return the full iterator
            #iter_start = self.start
        else:
            # in a worker process
            raise Exception('TODO: Implement multi-worker generator')
            # TODO: Implement later
            # split workload
            # per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            # worker_id = worker_info.id
            # iter_start = self.start + worker_id * per_worker
            # iter_end = min(iter_start + per_worker, self.end)

        return self.generator()



# ==================================== OLD =====================================
# def generator(char_img, epoch_size = 100, batch_size = 2):
#     '''
#     char_img   ---  images of characters
#                     these images should be separate for train/test sets
#     epoch_size ---  number of batches in epoch
#     '''

#     for N_ in range(epoch_size):
#         y_gt = [] #np.zeros((batch_size, n_chars_dict, n_chars))
#         imgs = []
#         for N in range(batch_size):
#             # stitching together the char-images to create an example
#             img, text = random_stamp_date(char_img)
#             imgs.append(np.copy(img))

#             # integers representation of text
#             y_gt.append([text_to_idx[t] for t in text])

#         # yield batch
#         imgs = torch.tensor(imgs)
#         y_gt = torch.tensor(y_gt)

#         yield imgs, y_gt
