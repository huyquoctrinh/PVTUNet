import tensorflow as tf 
import cv2 
import os 
from glob import glob
# from metrics.metrics_last import  iou_metric, MAE, WFbetaMetric, SMeasure, Emeasure,  dice_coef, iou_metric
import os 
import tensorflow as tf 
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss
from dataloader import build_augmenter, build_dataset, build_decoder
from tensorflow.keras.utils import get_custom_objects
# from model_research import build_student_model, create_segment_model
from model import build_model
# from supervision_model import build_student_model
import cv2 
import numpy as np
from model import build_model
# from distill_trainer import Distiller
# from save_model.best_model33 import build_model
# from save_model.ca_best_msf import build_model
# from save_model.best_up_ca import build_model
from tqdm import tqdm
# from model import build_model_caUnet, build_rivf_model
# from model_research import create_segment_model
import matplotlib.pyplot as plt 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

img_size = 352
BATCH_SIZE = 1
SEED = 1024
save_path = "/root/tqhuy/PolypPVT/best_model_pvt.h5"
route_data = "/root/tqhuy/torch_ver/dataset/TestDataset/"
# route_data = "/root/tqhuy/TVMI3K/Test/"
get_custom_objects().update({"dice": dice_loss})
model = build_model(img_size)
model.summary()
model.load_weights(save_path)

# model = create_segment_model(1)
# model = Distiller(img_size)
# model = Distiller(student=student_model, teacher=teacher_model)
# model.built = True
model.load_weights(save_path)
model.summary()

def load_dataset(route):
    BATCH_SIZE = 1
    X_path = '{}/images/'.format(route)
    Y_path = '{}/masks/'.format(route)
    X_full = sorted(os.listdir(f'{route}/images'))
    Y_full = sorted(os.listdir(f'{route}/masks'))

    X_train = [X_path + x for x in X_full]
    Y_train = [Y_path + x for x in Y_full]

    test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
    test_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=test_decoder, 
                                augmentAdv=False, augment=False, augmentAdvSeg=False, shuffle = None)
    return test_dataset, X_train

def predict(model, root, outdir):
    dataset, data_path = load_dataset(root)
    steps_per_epoch = len(data_path)//1
    # _, _, _, masks = model.predict(dataset, steps=steps_per_epoch)
    masks = model.predict(dataset, steps=steps_per_epoch)
    # if (smaller_masks is not None):
    #     smaller_masks_dir = outdir + "_small_mask/"
    #     if not os.path.exists(smaller_masks_dir):
    #         os.mkdir(smaller_masks_dir)
    print(masks.shape)
    i = 0
    for x, y in dataset:
        if (i == len(data_path)):
            break
        # print(y[0].shape)
        name = data_path[i].split("/")[-1].split(".")[0]
        # print(i, masks[i].shape)
        a = masks[i]
        mask_new = np.dstack([a, a, a])
        # small_mask = np.dstack([smaller_masks[i], smaller_masks[i], smaller_masks[i]])
        # print(x.shape, y.shape)
        # gt = np.dstack([y[0], y[0], y[0]])
        # gt = cv2.cvtColor(y[0], cv2.COLOR_GRAY2RGB)
        # true = cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB)
        # im_h = np.concatenate([x[0], gt * 255, mask_new *255], axis = 1)
        # cv2.imwrite()
        cv2.imwrite("{}/{}.png".format(outdir, name), mask_new*255)
        # if smaller_masks_dir is not None:
        #     # print(smaller_masks_dir)
        #     cv2.imwrite("{}/{}.png".format(smaller_masks_dir, name), small_mask*255)
        i+=1

# def pred_test(model, paths, save_dir = "./caformer/Kvasir"):
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    
#     for image_dir in paths:
#         img = read_image(image_dir)
#         # print(img.shape)
#         mask = model.predict(img)
#         mask = np.dstack([mask, mask, mask]) * 255.0
#         # print(image_dir)
#         name = image_dir.split("/")[-1].split(".")[0]
#         # print(name)
#         save_dir_img = save_dir + "/" + name +".png"
#         print(save_dir_img)
#         # mask = cv2.resize(mask, (626, 546))
#         cv2.imwrite(save_dir_img, mask)

def pred_full(model, root, dst = "./caformer/"):

    # predict(model, root, dst)
    ids = os.listdir(root)
    for id_data in ids:
        if not os.path.exists(dst + id_data):
            os.mkdir(dst + id_data)
        path = root + id_data+"/"
        predict(model, path, dst  + id_data)
        # test_x, test_y = load_data(path)
        # pred_test(model, test_x, dst + id_data)


# root = "/root/tqhuy/torch_ver/dataset/TestDataset/"
root = "/root/tqhuy/torch_ver/dataset/TestDataset/"
pred_full(model, root, dst = "/root/tqhuy/PolypPVT/save_vis/test_pvt/")

# path = "./TestDataset/Kvasir/"
# test_x, test_y = load_data(path)
# pred_test(model, test_x)
