import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/home/liyongjing/Egolee_2021/programs/yolov5-master_4/local_files")
from yolo_onnx_infer import YoloDetectionOnnx
from reid_onnx_infer import DeepPersonReidOnnx


class LockPersonInfer():
    def __init__(self, reid_onnx):
        self.reid_model = DeepPersonReidOnnx(reid_onnx, need_normailize=True)
        self.lock = False
        self.lock_img = None
        self.lock_feat = None
        self.lock_feat_online = None

    def set_lock_person(self, cv_img):
        self.lock_img = cv_img
        self.lock_feat = self.reid_model.infer_cv_img(cv_img)
        self.lock_feat_online = self.lock_feat
        self.lock = True
        cv2.namedWindow("lock_person", 0)
        cv2.imshow("lock_person", cv_img)
        print("set lock person")

    def get_lock_person(self, img, det_boxes):
        all_score_lock = []
        all_score_lock_online = []
        all_feats = []
        if self.lock:
            for det_box in det_boxes:
                p_img = img[int(det_box[1]):int(det_box[3]), int(det_box[0]):int(det_box[2]), :]
                if 0 not in p_img.shape:
                    p_feat = self.reid_model.infer_cv_img(p_img)

                    score_lock = float(np.matmul(self.lock_feat, p_feat.T))
                    score_lock_online = float(np.matmul(self.lock_feat_online, p_feat.T))

                    all_score_lock.append(score_lock)
                    all_score_lock_online.append(score_lock_online)
                    all_feats.append(p_feat)
                else:
                    all_score_lock.append(0)
                    all_score_lock_online.append(0)
                    all_feats.append(self.lock_feat_online)

        else:
            if len(det_boxes) > 0:
                sorted(det_boxes, key=lambda x: (x[1] - x[3]) * (x[0] - x[2]))
                det_box = det_boxes[0]
                p_img = img[int(det_box[1]):int(det_box[3]), int(det_box[0]):int(det_box[2]), :]
                if 0 not in p_img.shape:
                    self.set_lock_person(p_img)

        lock_result = None
        if len(all_score_lock) > 0:
            max_dist_lock = max(all_score_lock)
            if max_dist_lock > 0.6:
                max_indx = all_score_lock.index(max_dist_lock)
                lock_result = det_boxes[max_indx]
                self.lock_feat_online = all_feats[max_indx]
                print("max_dist_lock:", max_dist_lock)

        if lock_result is None:
            if len(all_score_lock_online) > 0:
                max_dist_lock_online = max(all_score_lock_online)
                if max_dist_lock_online > 0.7:
                    max_indx = all_score_lock_online.index(max_dist_lock_online)
                    lock_result = det_boxes[max_indx]
                    self.lock_feat_online = all_feats[max_indx]
                    print("max_dist_lock_online:", max_dist_lock_online)

        return lock_result



def test_lock_person_infer():
    reid_onnx_path = "/home/liyongjing/Egolee/programs/GoroboAIReason/models/person_lock/person_reid_v2.onnx"
    # reid_onnx_path = "/home/liyongjing/Egolee/programs/GoroboAIReason/models/person_lock/person_reid.onnx"
    lock_person_infer = LockPersonInfer(reid_onnx_path)

    onnx_path = '/home/liyongjing/Egolee_2021/programs/yolov5-master_4/runs/train/train_person_part_0304/weights/best.onnx'
    batch_size = 1
    yolo_detector = YoloDetectionOnnx(onnx_path, batch_size)
    confidents = np.array([0.6])
    yolo_detector.batch_size = batch_size
    yolo_detector.min_conf = confidents[0]

    # image_path = "/home/liyongjing/Egolee_2021/data/open_dataset/pose_track/images/val/024159_mpii_test"

    image_path = "/home/liyongjing/Egolee_2021/data/src_track/1-20210510_15-00-16_cut"
    image_names = list(filter(lambda x: x[-3:] == "jpg" and int(x.split('.')[0])%5==0, os.listdir(image_path)))
    image_names.sort(key=lambda x: int(x.split(".")[0]))
    for img_name in tqdm(image_names):
        print(img_name)
        img = cv2.imread(os.path.join(image_path, img_name))
        img_show = img.copy()
        det_boxes = yolo_detector.infer_cv_img(img)
        for det_box in det_boxes:
            box = det_box[0:4]
            score = det_box[4:5]
            cls = det_box[5:6]
            show_color = (0, 255, 0)

            # size filter
            h, w, _ = img_show.shape
            box_w = int(box[2] - box[0])
            box_h = int(box[3] - box[1])
            if box_w/w > 0.06 or box_h/w > 0.06:
                cv2.rectangle(img_show, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), show_color, 2)
                cv2.putText(img_show, str(round(score[0], 2)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, show_color, 1)

        lock_result = lock_person_infer.get_lock_person(img, det_boxes)
        if lock_result is not None:
            cv2.rectangle(img_show, (int(lock_result[0]), int(lock_result[1])), (int(lock_result[2]), int(lock_result[3])), (0, 0, 255), 2)

        cv2.namedWindow("img_show", 0)
        cv2.imshow("img_show", img_show)
        wait_key = cv2.waitKey(0)
        if wait_key == 27:
            exit(1)





if __name__ == "__main__":
    test_lock_person_infer()
