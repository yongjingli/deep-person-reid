import sys
sys.path.insert(0, "../")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import cv2
import onnxruntime
import numpy as np
import os


class DeepPersonReidOnnx():
    def __init__(self, onnx_path, need_normailize=True):
        self.onnx_path = onnx_path
        self.ort_sess = onnxruntime.InferenceSession(self.onnx_path)
        self.input_name = self.ort_sess.get_inputs()[0].name

        self.dst_w = 128
        self.dst_h = 256
        self.input_size = [self.dst_h, self.dst_w]

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.std_inv = 1 / self.std
        self.img_t = None
        self.need_normailize = need_normailize

    def img_preprocess(self, cv_img):
        cv_img = cv2.resize(cv_img, (self.dst_w, self.dst_h), cv2.INTER_LINEAR)
        cv_img = cv_img.copy().astype(np.float32)
        cv_img = cv_img/255
        mean = np.float64([0.485, 0.456, 0.406]).reshape(1, -1)
        std = np.float64([0.229, 0.224, 0.225]).reshape(1, -1)
        std_inv = 1 / np.float64(std.reshape(1, -1))

        cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB, cv_img)
        cv2.subtract(cv_img, mean, cv_img)
        cv2.multiply(cv_img, std_inv, cv_img)

        cv_img = cv_img.transpose(2, 0, 1)  # to C, H, W
        cv_img = np.ascontiguousarray(cv_img)
        cv_img = np.expand_dims(cv_img, axis=0)
        return cv_img

    def infer_cv_img(self, cv_img):
        self.img_t = self.img_preprocess(cv_img)
        feature = self.ort_sess.run(None, {self.input_name: self.img_t})[0]
        if self.need_normailize:
            feature = self.normalize(feature)
        return feature

    def normalize(self, nparray, order=2, axis=-1):
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)


def deep_person_reid_infer():
    onnx_path = "/home/liyongjing/Egolee_2021/programs/deep-person-reid-master/local_files/osnet_ain_x1_0.onnx"
    images_dir = '/home/liyongjing/Egolee_2021/data/open_dataset/market1501/Market-1501-v15.09.15/gt_bbox'
    deep_person_reid_onnx = DeepPersonReidOnnx(onnx_path)
    image_names = [f for f in os.listdir(images_dir) if os.path.splitext(f)[-1] in ['.jpg', 'png']]
    for image_name in image_names:
        img_path = os.path.join(images_dir, image_name)
        img_path = "/home/liyongjing/Egolee_2021/data/open_dataset/market1501/Market-1501-v15.09.15/query/0632_c3s2_024837_00.jpg"
        img = cv2.imread(img_path)
        feature = deep_person_reid_onnx.infer_cv_img(img)
        print(feature)
        exit(1)


def test_query_person():
    onnx_path = "/home/liyongjing/Egolee_2021/programs/deep-person-reid-master/local_files/osnet_ain_x1_0.onnx"
    deep_person_reid_onnx = DeepPersonReidOnnx(onnx_path)
    q_img = cv2.imread("/home/liyongjing/Egolee_2021/data/open_dataset/market1501/Market-1501-v15.09.15/query/0001_c3s1_000551_00.jpg")
    q_feat = deep_person_reid_onnx.infer_cv_img(q_img)


    images_dir = "/home/liyongjing/Egolee_2021/data/open_dataset/market1501/Market-1501-v15.09.15/bounding_box_test"
    image_names = [f for f in os.listdir(images_dir) if os.path.splitext(f)[-1] in ['.jpg', 'png']]
    for image_name in image_names:
        img_path = os.path.join(images_dir, image_name)
        g_img = cv2.imread(img_path)
        g_feat = deep_person_reid_onnx.infer_cv_img(g_img)
        feat_score = float(np.matmul(q_feat, g_feat.T))
        print(feat_score)

        cv2.imshow("q_img", q_img)
        cv2.imshow("g_feat", g_img)

        h, w, _ = g_img.shape
        center = (int(w/2), int(h/2))
        cv2.putText(g_feat, str(round(feat_score, 2)), center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

        if feat_score > 0.5:
            wait_key = cv2.waitKey(0)
        else:
            wait_key = cv2.waitKey(1)

        if wait_key == 27:
            exit(1)



if __name__ == "__main__":
    logger.info("Start Proc...")
    # deep_person_reid_infer()
    test_query_person()
    logger.info("End Proc...")
