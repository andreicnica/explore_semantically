"""
    Code source https://github.com/amdegroot/ssd.pytorch
"""

import os.path as osp
import sys
import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import cv2
import numpy as np

COCO_ROOT = osp.join("dataset")
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES_2017 = \
    ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
     'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
     'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
     'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
     'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    batch = zip(*batch)  # Group by type
    imgs, boxes, segmentations = batch
    imgs = [torch.from_numpy(np.stack(x)) for x in zip(*imgs)]
    segmentations = [torch.from_numpy(np.stack(x)) for x in zip(*segmentations)]
    boxes = [torch.from_numpy(x) for x in boxes]

    return imgs, boxes, segmentations


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:

                bbox = obj['bbox']
                bbox = np.array(bbox).astype(np.float32)
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]

                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(bbox/scale)
                if (np.array(final_box) > 1.).any():
                    print("WTFFFFFFF")

                final_box.append(label_idx)

                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set='val2017', transform=None,
                 target_transform=COCOAnnotationTransform(), dataset_name='MS COCO'):
        sys.path.append(osp.join(root, COCO_API))
        self.root = osp.join(root, IMAGES, image_set)
        self.coco = COCO(osp.join(root, ANNOTATIONS,
                                  INSTANCES_SET.format(image_set)))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, sg, h, w, img_id = self.pull_item(index)
        return im + sg, img_id

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        target = self.coco.loadAnns(ann_ids)
        path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(osp.join(self.root, path))
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target, dtype=np.float32)
            imgs, boxes, labels, segmentations = self.transform(img, target[:, :4], target[:, 4])

            # to rgb
            imgs = [img[:, :, (2, 1, 0)] for img in imgs]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return [torch.from_numpy(img).permute(2, 0, 1) for img in imgs], target, segmentations, \
               height, width, img_id

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def remake(img, mean, box=None):
    img = img.numpy().copy()
    img = img.transpose((1, 2, 0))
    img = img[:, :, (2, 1, 0)]
    img += np.array(mean)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8).copy()
    if box is not None:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255))
    return img


if __name__ == "__main__":
    from utils.ssd_augmentation import SSDAugmentation

    dataset_root = "/raid/workspace/alexandrug/coco/coco"
    image_set = "train2017"
    data_mean = [104, 117, 123]
    dim = [256, 128]

    dataset = COCODetection(root=dataset_root, image_set=image_set,
                            transform=SSDAugmentation(dim, data_mean))
    train_loader = data.DataLoader(dataset, 32,
                                   num_workers=40,
                                   shuffle=False,  # collate_fn=detection_collate,
                                   pin_memory=True)
    for batch_idx, data in enumerate(train_loader):
        #sg_scale0 = sg[0]
        #for bidx in range(gt.shape[0]):
        #    class_idx = int((gt[bidx])[-1])

            #print(COCO_CLASSES_2017[class_idx], gt[bidx])

        #   img = remake(im[0], data_mean, box=((gt[bidx][:4]) * dim[0]).astype(np.int))
            #cv2.imshow("IMG", img)
            # img_small = cv2.resize(img, (16, 16))
            # cv2.imshow("IMG_small", img_small)
            # img_small = cv2.resize(img_small, (256, 256))
            # cv2.imshow("IMG_resized", img_small)
            #cv2.imshow("Class Segm", sg_scale0[class_idx])
            #cv2.waitKey(0)
        print("-" * 100)
        print("Batch: " + str(batch_idx))
    # #
    # segmentation = torch.zeros(80, *im[0].size()[1:])
    #
    #
    # print(im[0].size())
    # print(im[1].size())
    # print(gt)
    # print(w, h)
    #
    # import cv2
    #
    # x = np.zeros((256, 256), dtype=np.uint8)
    # x[100:150, 100:200] = 255
    #
    # blur = cv2.GaussianBlur(x,(129*2+1, 129), 0)
    # cv2.normalize(blur, blur, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow("test1", blur)
    # cv2.waitKey(0)
    # #
    # for i in range(5):
    #     blur = cv2.GaussianBlur(blur,(129*2+1, 129), 0)
    #     cv2.normalize(blur, blur, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow("test2", blur)
    # cv2.waitKey(0)
    #
    # blur = cv2.medianBlur(blur,(129,129),0)
    # # blur = cv2.GaussianBlur(blur,(51,51),0)
    # # blur = cv2.GaussianBlur(blur,(51,51),0)
    #
    # cv2.imshow("test2", blur)
    # cv2.waitKey(0)
    #
    #
    # import numpy as np
    #
    #
    # def softmax(x, temp=1.):
    #     """Compute softmax values for each sets of scores in x."""
    #     return np.exp(x/temp) / np.sum(np.exp(x/temp))
    #
    #
    # def makeGaussian(size, fwhm = 3, center=None, x=None, y=None):
    #     """ Make a square gaussian kernel.
    #
    #     size is the length of a side of the square
    #     fwhm is full-width-half-maximum, which
    #     can be thought of as an effective radius.
    #     """
    #
    #     if x is None:
    #         x = np.arange(0, size, 1, float)
    #         y = x[:, np.newaxis]
    #
    #     if center is None:
    #         x0 = y0 = size // 2
    #     else:
    #         x0 = center[0]
    #         y0 = center[1]
    #
    #     return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    #     # return 7**(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    #
    #
    #
    # size=256
    #
    # old_s = np.zeros((size, size))
    # for i in range(100):
    #     yt = np.random.randint(0, size-1)
    #     yb = np.random.randint(yt+1, size)
    #     xl = np.random.randint(0, size-1)
    #     xr = np.random.randint(xl+1, size)
    #     yt = 50+50
    #     yb = 150+50
    #     xl = 50+50
    #     xr = 155+50
    #
    #     img = np.zeros((size, size), dtype=np.uint8)
    #     height = yb - yt
    #     width = xr - xl
    #
    #     arx = ary = 1.
    #     if width < height:
    #         ary = (width / float(height) + 1.) / 2.
    #     else:
    #         arx = (float(height) / width + 1.) / 2.
    #
    #     # ar = 1
    #     cy = (yt + yb) / 2 * ary
    #     cx = (xl + xr) / 2 * arx
    #
    #     img[yt:yb, xl:xr] = 255
    #     # x = makeGaussian(x, y, 256, fwhm=256)
    #     x = np.arange(0, size, 1, float) * arx
    #     y = np.arange(0, size, 1, float)[:, np.newaxis] * ary
    #
    #     # %timeit for x in range(100): makeGaussian(256, fwhm=256, x=x, y=y)
    #     fwhm = max(cx, cy, size-cx, size-cy)
    #     # fwhm = max(width, height) * 15
    #     # fwhm = 5
    #     temp = (((width * height) / (size ** 2)) ** 0.5) * 2
    #     gauss = makeGaussian(size, fwhm=fwhm, center=(cx, cy), x=x, y=y)
    #     s = softmax(gauss, temp=temp)
    #     s *= 1./s.max()
    #
    #     cv2.imshow("test4", s_1)
    #     cv2.waitKey(0)
    #     cv2.imshow("test5", s)
    #     cv2.waitKey(0)
    #
    #     old_s = np.max(np.stack([s_1, s]), axis=0)
    #
    #     cv2.imshow("old", old_s)
    #     cv2.waitKey(0)
    # #
