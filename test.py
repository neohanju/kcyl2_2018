import cv2
import math
import numpy as np
import numpy.matlib
import glob
import os
import argparse
import preprocessing as pp
from numpy.linalg import inv
from numpy import linalg as LA
import enum
from skimage import io
import face_alignment

class Alignment(enum.Enum):
    TIGHT = 0
    LOOSE = 1
    NOT_ALIGNED = 2

class CNN(enum.Enum):
    TIGHT = 0
    LOOSE_SMALL = 1
    LOOSE_BIG = 2

# kCelebAEyeL = [68, 111]
# kCelebAEyeR = [107, 112]
# kCelebAWidth = 178
# kCelebAHeight = 218

kPathSize = [64, 224]
kEyeLooseL = np.float32([0.5-1/8, 1/2])
kEyeLooseR = np.float32([0.5+1/8, 1/2])
kEyeLooseDistance = LA.norm(kEyeLooseR - kEyeLooseL)
kEyeTightL = np.float32([20/64, 33/64])
kEyeTightR = np.float32([42/64, 33/64])
kEyeTightDistance = LA.norm(kEyeTightR - kEyeTightL)

kAlignMargin = 5/64
kAllowablePaddingRatio = 0.05

kResultFilePostfix = 'K-CYL2_김태훈'

# ======================================================================================================================
# Options
# ======================================================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='directory containing input test data')
parser.add_argument('--result_dir', type=str, help='save path')
parser.add_argument('--model_dir', type=str, default='models', help='save path')
options = parser.parse_args()
print(options)


def save_result_to_txt(_result_dict_list, _file_path):
    with open(_file_path, "w") as res_file:
        for cur_dict in _result_dict_list:
            res_file.writelines(cur_dict['problem_no'] + ",%1.6f" % cur_dict['prob'])


def do_mission_1(_data_dir, _res_dir, _face_landmark_detector):
    # =========================================================================
    # MISSION 1
    # =========================================================================
    # model load

    # model_torso_hq = ... os.path.join(options.model_dir, 'mission1_torso_hq.pth)
    # model_torso_lq = ... os.path.join(options.model_dir, 'mission1_torso_lq.pth)
    # model_face = ... os.path.join(options.model_dir, 'mission1_face.pth)

    # file load
    file_name_list = glob.glob(_data_dir + '/1_*.*')
    prediction_results = []
    for file_name in file_name_list:
        img = io.imread(file_name)
        problem_number = os.path.basename(file_name).split('.')[0]
        cur_result = {'problem_no': problem_number, 'prob': 1.0}  # <= default value is one to handle landmark missing

        # landmark prediction
        preds = _face_landmark_detector.get_landmarks(input)
        if preds is None:
            prediction_results.append(cur_result)
            continue

        # image alignment
        left_eye = [preds[36:42, 0].mean(), preds[36:42, 1].mean()]
        right_eye = [preds[42:48, 0].mean(), preds[42:48, 1].mean()]
        img_aligned, alignment_type = pp.align_face_image(img, [left_eye, right_eye])

        if alignment_type == Alignment.TIGHT:
            print('tight')
            cur_result['prob'] = 1.0
            # todo Implement tight
        elif alignment_type == Alignment.LOOSE:
            print('loose')
            cur_result['prob'] = 1.0
            # todo Implement loose
            # if  shape is 64x64 = > LQ
            # elif shape is 224x224 => HQ

        prediction_results.append(cur_result)

    # save result
    save_result_to_txt(prediction_results, os.path.join(_res_dir, 'mission1_%s.txt' % kResultFilePostfix))


def do_mission_2(_data_dir, _res_dir, _face_landmark_detector):
    # =========================================================================
    # MISSION 2
    # =========================================================================
    file_name_list = glob.glob(_data_dir + '/2_*.*')
    prediction_results = []
    for file_name in file_name_list:
        img = io.imread(file_name)
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)

        problem_number = os.path.basename(file_name).split('.')[0]
        cur_result = {'problem_no': problem_number, 'prob': 1.0}  # <= default value is one to handle landmark missing

        print(problem_number)
        preds_allfaces = fa.get_landmarks(img, all_faces=True)

        max_prob = 0.0
        for preds_face in preds_allfaces:
            # align per face
            left_eye = [preds_face[36:42, 0].mean(), preds_face[36:42, 1].mean()]
            right_eye = [preds_face[42:48, 0].mean(), preds_face[42:48, 1].mean()]
            img_aligned, alignment_type = pp.align_loose_image(img, [left_eye, right_eye])

            # todo : save image for test
            Image.fromarray(img_aligned).save('%s/%06d.png' % (_data_dir, i))
            i += 1
            # todo : or feed to vgg16
            # todo : get result
            cur_prob = 0.4  # temporal value
            if max_prob < cur_prob:
                max_prob = cur_prob

        cur_result['prob'] = max_prob
        prediction_results.append(cur_result)

    # save result
    save_result_to_txt(prediction_results, os.path.join(_res_dir, 'mission2_%s.txt' % kResultFilePostfix))


if __name__ == "__main__":

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=True)

    do_mission_1(options.data_dir, options.result_dir, fa)
    do_mission_2(options.data_dir, options.result_dir, fa)

# ()()
# ('') HAANJU & YEOLJERRY





