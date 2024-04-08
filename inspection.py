
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,
                                  TextArea)
import h5py
# spatial indexing:
from rtree import index

import os
import pdb, ipdb

from gui_utils import select_file
from filepaths import imdb_dir, images_dir, h5_path

# # manually intercept argv to supply debug args:
# sys.argv = ['train.py', '-d', 'PFL-DocVQA-local', '-m', 'VT5', ]# '--use_dp']


FIG_SIZE = (6,8)

ALWAYS_SHOW_BBOXES = True


####################
# main visualisation function:
###################

cur_event = None
prev_artist_ids = []

def display_record(img_path, words, bboxes, qa_pairs, use_h5=False, h5_path=None):

    # initialise global variables for mouseover hook:
    global cur_event
    cur_event = None
    global prev_artist_ids
    prev_artist_ids = []

    # get the image:
    # img_path = os.path.join(images_dir, f"{img_name}.jpg")

    if not use_h5:
        # loading a normal jpeg
        img = Image.open(img_path).convert("RGB")
        print(f'Displaying: {img_path}')

    else:
        assert h5_path is not None
        # loading a serialised h5 file
        # the img path is after the h5 separator:
        img_filename = img_path.split('/')[-1]
        img_prefix = img_filename.split('.')[0]

        h5_file = h5py.File(h5_path, 'r')
        img_arr = h5_file[img_prefix][:]
        img = Image.fromarray(img_arr)
        print(f'Displaying: {h5_path}/{img_prefix}')

    img_w, img_h = (img.size)

    print(f'  dims (WxH): {img_w}x{img_h}')


    # print all question-answer pairs to console:
    for pair in qa_pairs:
        question, answer = pair
        print(f'    Q: {question}')
        ans_lines = answer.split('\n')
        print(f'      A: {ans_lines[0]}')
        # rest of lines have same offset:
        for ans_line in ans_lines[1:]:
            print(f'         {ans_line}')

    # plot the image:

    fig = plt.figure(figsize=FIG_SIZE) # a4 aspect ratio (ish)
    # ax1 = fig.add_subplot(111)
    ax1 = fig.gca()
    axim = ax1.imshow(img)

    # create spatial index:
    spindex = index.Index()
    # and indexed lookups for artist objects:
    artists = []
    for i, bbox in enumerate(bboxes):
        l1,t1,r1,b1 = bbox

        # convert to pixel units:
        l2, r2 = [int(c * img_w) for c in (l1, r1)]
        t2, b2 = [int(c * img_h) for c in (t1, b1)]

        # insert the bbox (in pixel units) to spatial index:
        spindex.insert(i, (l2,t2,r2,b2))

        # get width and height:
        bbox_w = r2-l2
        bbox_h = t2-b2
        # and bbox centre coords:
        bbox_cx = l2 + (bbox_w // 2)
        bbox_cy = b2 + (bbox_h // 2)

        # draw the bounding box:
        rect = Rectangle((l2,b2), bbox_w, bbox_h,
                         fill=False, ec='tab:orange', alpha=0.5)

        vert_anchor = t2
        vert_offset_frac = 0.05

        # if t1 < 0.1: # box is near the top, so annotate below
        #     vert_offset_frac = -0.05
        #     vert_anchor = b2 # drawing from bottom of bbox
        # else: # annotate above
        #     vert_offset_frac = 0.05
        #     vert_anchor = t2 # drawing from top
        #
        # # offset by fraction of current axis ylim:
        # cur_ylim = ax1.get_ylim()
        # cur_ax_height = cur_ylim[0] - cur_ylim[1]
        #
        # vert_offset_px = vert_offset_frac * cur_ax_height

        # annotate with the corresponding string:
        labelbox = TextArea(words[i], textprops=dict(color='white', horizontalalignment='center'))
        anno = AnnotationBbox(labelbox, xy=(bbox_cx, vert_anchor),
                            xybox=(bbox_cx, vert_anchor),#  - vert_offset_px),
                            xycoords='data',
                            boxcoords='data',
                            arrowprops=dict(linestyle="--", lw=1, fill=False, arrowstyle='-', color='tab:orange'),
                            bboxprops=dict(linestyle="--", edgecolor='tab:orange', facecolor='black', alpha=0.6),
                            )
        # add to figure:
        ax1.add_artist(rect)
        ax1.add_artist(anno)

        # but hide at first:
        anno.set_visible(False)
        if not ALWAYS_SHOW_BBOXES:
            rect.set_visible(False)

        # and store for lookup later:
        artists.append((rect, anno,))

    # now, when we hover over a part of the image, display the corresponding artist:
    def hover(event):

        # get the list of artist IDs that intersect with the hover point:
        point_bbox = (event.xdata, event.ydata, event.xdata, event.ydata)
        if event.inaxes is not None:
            intersections = list(spindex.intersection(point_bbox))
            global cur_event
            cur_event = event

        else:
            intersections = []

        # store which artists are currently visible, so we can hide them later:
        global prev_artist_ids

        # display all intersecting artists:
        if len(intersections) > 0:
            for i in intersections:
                rect, anno = artists[i]

                ### dynamically set annotation distance:
                cur_xy = anno.xy
                cur_ylim = ax1.get_ylim()
                cur_ax_height = cur_ylim[0] - cur_ylim[1]

                vert_offset_px = vert_offset_frac * cur_ax_height
                anno.xybox = (cur_xy[0], cur_xy[1] - vert_offset_px)
                ###

                anno.set_visible(True)
                if not ALWAYS_SHOW_BBOXES:
                    rect.set_visible(True)

        # hide previous artists if they are not intersecting:
        for i in prev_artist_ids:
            if i not in intersections:
                rect, anno = artists[i]
                anno.set_visible(False)
                if not ALWAYS_SHOW_BBOXES:
                    rect.set_visible(False)
        prev_artist_ids.clear()

        # current intersections are the next prev_artist_ids
        prev_artist_ids.extend(intersections)

        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.show()

def squeeze(x):
    # if x is a list of length 1, returns the contained item as a bare object
    if isinstance(x, list):
        if len(x) == 1:
            return x[0]
        else:
            raise Exception(f'squeezed item is a list longer than 1: {x}')
    else:
        return x

def inspect_batch(batch, max_samples=None):
    # given a PFL-DocVQA data batch or similar,
    # load the image and OCR annotations etc. of every example in the batch
    img_paths = batch['image_names']
    batch_size = len(img_paths)

    # mapping from batch keys to record keys:
    mapping = {'image_names': 'image_name',
               'questions': 'question',
                'words': 'ocr_tokens',
                'boxes': 'ocr_normalised_boxes',
    }
    pseudorecords = [{key: value[b] for key,value in batch.items()} for b in range(batch_size)]
    for rec in pseudorecords[:max_samples]:
        inspect_record(rec)


def inspect_record(record, use_h5=False, h5_path=None):
    # given a PFL-DocVQA record or similar,
    # load the image and its OCR annotations directly
    # img_path = squeeze(record['image_names'])
    img_path = record['image_names']

    # words, bboxes = squeeze(record['words']), squeeze(record['boxes'])
    words, bboxes = record['words'], record['boxes']
    # qa_pairs = [[squeeze(record['questions']), squeeze(squeeze(record['answers']))]]
    qa_pairs = [[record['questions'], record['answers'][0]]]

    if 'label_name' in record.keys():
        label_name = record['label_name']
        print(f'{label_name=}')

    display_record(img_path, words, bboxes, qa_pairs, use_h5, h5_path)

def inspect_img(img_name, images_dir, img_ocrs, img_questions, use_h5=False, h5_path=None):
    # given the name of an image, and the pre-loaded OCR annotations etc.,
    # display it and its bounding boxes

    # add directory listing if not present:
    if images_dir not in img_name:
        img_path = os.path.join(images_dir, img_name)
    else:
        img_name = '/'.split(img_name)[-1]
        img_path = img_name

    # add filename extension if not present:
    if '.jpg' not in img_name:
        img_prefix = img_name
        img_filename = img_name + '.jpg'
        img_path = img_path + '.jpg'
    else:
        img_prefix = img_name.split('.jpg')[0]
        img_filename = img_name

    # and its OCR data:
    words, bboxes = img_ocrs[img_prefix]

    # get all Q&A pairs associated with this image:
    qa_pairs = img_questions[img_prefix]

    display_record(img_path, words, bboxes, qa_pairs, use_h5, h5_path)





def main():

    print(f'Loading imdb...')

    centralized = True
    v1 = True
    use_h5 = True

    if v1:
        prefix = 'blue'
        val_name = 'valid'
    else:
        prefix = 'imdb'
        val_name = 'val'

    if (v1) or (not centralized):
        imdb_files = [f"{prefix}_train_client_{i}.npy" for i in range(10)]
    else:
        imdb_files = [f"{prefix}_train.npy"]
    imdb_files.append(f"{prefix}_{val_name}.npy")
    imdb_files.append(f"{prefix}_test.npy")

    all_data = []
    all_imdb = []

    # # load data if it's not loaded already:
    # if 'data' not in globals().keys():
    for filename in imdb_files:
        print(f'Loading data record {filename}...')
        try:
            data = np.load(os.path.join(imdb_dir, filename), allow_pickle=True)
            header = data[0]
            imdb = data[1:]
            all_data.append(data)
            all_imdb.append(imdb)
        except Exception as e:
            print(f'Error: {e}')

    data = np.concatenate(all_data, axis=0)
    imdb = np.concatenate(all_imdb, axis=0)
    del all_data, all_imdb

    # loop through all questions, allocate them to image names:
    img_ocrs = {}
    img_questions = {}

    for record in imdb:
        img_name = record['image_name']
        if img_name not in img_questions:
            # initialise empty list of questions linked to this image
            img_questions[img_name] = []
            # and store the ocr tokens/boxes: (which are the same for each image)
            ocr_pair = record['ocr_tokens'], record['ocr_normalized_boxes']
            img_ocrs[img_name] = ocr_pair

        # store this question, answer pair:
        qa_pair = record['question'], record['answers']
        img_questions[img_name].append(qa_pair)


    # main interactive loop:
    while True:

        # bring up a file chooser and return the chosen filepath:
        selection = select_file(images_dir)
        if selection is None:
            print('Finished, closing.')
            break
        else:
            # last part of the filepath is the img name:
            img_name_ext = selection.split('/')[-1]
            # strip the .jpg extension:
            img_name = img_name_ext.split('.jpg')[0]

        inspect_img(img_name, images_dir, img_ocrs, img_questions, use_h5=use_h5, h5_path=h5_path)

if __name__ == '__main__':
    main()
