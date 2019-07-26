import cv2
import numpy as np
import torch
import torchvision
import pandas as pd

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def remake(img, mean, std, box=None):
    img = img.numpy().copy()
    img = img.transpose((1, 2, 0))
    img = img[:, :, (2, 1, 0)]
    img *= std
    img += mean
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8).copy()
    if box is not None:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255))
    return img


def plot_train():
    file = "results/2019Jul19-131320_default_base/0000_default_base/0/log.csv"
    df = pd.read_csv(file, skiprows=[0], header=None)
    df.columns = ["iter_no", "epoch", "batch_idx", "bps", "gradientμ", "gradientstd", "gradientmax", "lossμ", "lossstd",
                  "lossmin", "lossmax", "loss_evalμ", "loss_evalstd", "loss_evalmin", "loss_evalmax"]
    df_train = df[~df.bps.isna()]
    df_eval = df[df.bps.isna()]

    max_eval_batch = df_eval.batch_idx.max()
    df_eval[df_eval.batch_idx == max_eval_batch].plot("epoch", "loss_evalμ")
    df_train.groupby("epoch")["lossμ"].mean().plot()
    df_train["lossμ"].plot()
    df_train


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Show images.')
    parser.add_argument('data')
    args = parser.parse_args()

    mean = np.array(MEAN) * 255.
    std = np.array(STD) * 255.

    data = torch.load(args.data)

    # Show image

    imgs = data["imgs"]
    segmentation = data["segmentation"][0]
    predict = data["predict"]

    print(f"No of batches: {imgs[0].size(0)}")

    for idx in range(imgs[0].size(0)):
        img = remake(imgs[0][idx], mean, std)

        cv2.imshow("IMG", img)
        # cv2.waitKey(0)

        seg = segmentation[idx]
        seg = seg / seg.max()
        grid = torchvision.utils.make_grid(seg.unsqueeze(1), nrow=8, padding=10, scale_each=0.2)
        grid_img = grid[0].numpy()
        grid_img = cv2.resize(grid_img, (0,0), fx=0.2, fy=0.2)

        cv2.imshow("Seg", grid_img)

        pred = predict[idx]

        # pred = torch.nn.Sigmoid()(pred)
        print(pred.size())
        pred = torch.nn.Softmax(dim=1)(pred)

        pred = pred.view(80, 256, 256)
        pred -= pred.min()
        pred = pred / pred.max()
        grid = torchvision.utils.make_grid(pred.unsqueeze(1), nrow=8, padding=10, scale_each=0.2)
        grid_img = grid[0].numpy()
        grid_img = cv2.resize(grid_img, (0,0), fx=0.2, fy=0.2)

        cv2.imshow("pred", grid_img)
        cv2.waitKey(0)


