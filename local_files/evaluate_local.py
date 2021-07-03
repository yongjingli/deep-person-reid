import os
import numpy as np
import torch
import cv2
import torchreid
from torchreid import metrics
from torch.nn import functional as F
from tqdm import tqdm


def evaluate_market_local():
    dataset_cfg = {'root': '/home/liyongjing/Egolee_2021/data/open_dataset',
                   'sources': ['market1501'], 'targets': ['market1501'],
                   'height': 256, 'width': 128, 'transforms': ['random_flip', 'color_jitter'],
                   'k_tfm': 1, 'norm_mean': [0.485, 0.456, 0.406], 'norm_std': [0.229, 0.224, 0.225],
                   'use_gpu': True, 'split_id': 0, 'combineall': False, 'load_train_targets': False,
                   'batch_size_train': 64, 'batch_size_test': 1, 'workers': 4, 'num_instances': 4, 'num_cams': 1,
                   'num_datasets': 1, 'train_sampler': 'RandomSampler', 'train_sampler_t': 'RandomSampler',
                   'cuhk03_labeled': False, 'cuhk03_classic_split': False, 'market1501_500k': False}

    datamanager = torchreid.data.ImageDataManager(**dataset_cfg)

    train_loader = datamanager.train_loader
    test_loader = datamanager.test_loader

    model = torchreid.models.build_model(name='osnet_ain_x1_0', num_classes=1041, pretrained=False)
    torchreid.utils.load_pretrained_weights(model, "/home/liyongjing/Egolee_2021/programs/deep-person-reid-master/weights/osnet_ain_d_m_c.pth.tar")
    model.cuda()
    model.eval()

    targets = list(test_loader.keys())

    for name in targets:
        domain = 'source' if name in datamanager.sources else 'target'
        print('##### Evaluating {} ({}) #####'.format(name, domain))
        query_loader = test_loader[name]['query']
        gallery_loader = test_loader[name]['gallery']

        def parse_data_for_eval(data):
            imgs = data['img']
            pids = data['pid']
            camids = data['camid']

            # impath = data['impath']
            # print(imgs)
            # print(impath)

            return imgs, pids, camids

        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = parse_data_for_eval(data)
                imgs = imgs.cuda()
                with torch.no_grad():
                    features = model(imgs)
                features = features.cpu().clone()
                # features = F.normalize(features, p=2, dim=1)
                # print(features)
                # exit(1)
                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_

        print('Extracting features from query set ...')
        qf, q_pids, q_camids = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        distmat = metrics.compute_distance_matrix(qf, gf, 'cosine')
        distmat = distmat.numpy()

        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=False
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        ranks = [1, 5, 10, 20]
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))



from PIL import Image
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip
)


def test_img_preprocess():
    img_path = "/home/liyongjing/Egolee_2021/data/open_dataset/market1501/Market-1501-v15.09.15/query/0632_c3s2_024837_00.jpg"
    cv_img = cv2.imread(img_path)

    # img = Image.open(img_path).convert('RGB')
    # cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    #

    cv_img = cv2.resize(cv_img, (128, 256), cv2.INTER_LINEAR)
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
    print(cv_img)

    img = Image.open(img_path).convert('RGB')
    norm_mean = [0.485, 0.456, 0.406] # imagenet mean
    norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    height, width = 256, 128

    transform_te = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    out = transform_te(img)
    out = out.unsqueeze(dim=0)
    print(out.numpy())

    np.set_printoptions(threshold=np.inf)
    diff = cv_img - out.numpy()
    print(diff)



def compare_cv_pil_read():
    img_path = "/home/liyongjing/Egolee_2021/data/open_dataset/market1501/Market-1501-v15.09.15/query/0632_c3s2_024837_00.jpg"
    cv_img = cv2.imread(img_path)
    # cv_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    img = Image.open(img_path).convert('RGB')
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    diff = cv_img - img

    cv2.imshow("cv_img", cv_img)
    cv2.imshow("img", img)
    cv2.waitKey(0)

    print(diff)


if __name__ == "__main__":
    evaluate_market_local()
    # test_img_preprocess()
    # compare_cv_pil_read()
