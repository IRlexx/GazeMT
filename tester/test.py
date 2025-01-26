import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse

def draw_arrow(image, start_point, end_point, color, thickness=2):
    
    image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=0.3)
    return image

def main(train, test, fold_to_test):

    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)

    data = test.data
    load = test.load

    print("===> Read data <===")

    if data.isFolder:
        data, _ = ctools.readfolder(data)

    
    label_files = data.label
    assert len(label_files) == 4, "Expected 4 label files for 3-fold cross-validation"
    print(label_files)

    
    assert 1 <= fold_to_test <= 3, "Fold number must be between 1 and 3"

    
    fold = fold_to_test - 1
    print(f"===> Testing Fold {fold_to_test} <===")

    
    test_file = label_files[fold]

   
    test_data = edict(data)
    test_data.label = [test_file]

    print(f"==> Test: {test_data.label} <==")
    dataset = reader.loader(test_data, 32, num_workers=4, shuffle=False)

    modelpath = os.path.join(train.save.metapath, train.save.folder, "checkpoint")
    logpath = os.path.join(train.save.metapath, train.save.folder, test.savename)

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Test <=============================

    for epoch in range(10, 81, 10): 
        model_file = f"Iter_{epoch}_{train.save.model_name}.pt"
        model_path = os.path.join(modelpath, f"fold_{fold+1}_checkpoint", model_file)
        print(f"Loading model from {model_path}")

        net = model.Model()

        statedict = torch.load(
            model_path,
            map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
        )

        net.cuda()
        net.load_state_dict(statedict)
        net.eval()

        length = len(dataset)
        accs = 0
        count = 0

        logname = f"fold_{fold+1}_epoch_{epoch}.log"

        outfile = open(os.path.join(logpath, logname), 'w')
        outfile.write("name results gts\n")

        with torch.no_grad():
            for j, (data, label) in enumerate(dataset):
                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                names = data["name"]
                gts = label.cuda()

                gazes = net(data)

                for k, gaze in enumerate(gazes):
                    gaze = gaze.cpu().detach().numpy()
                    gt = gts.cpu().numpy()[k]

                    count += 1
                    accs += gtools.angular(
                        gtools.gazeto3d(gaze),
                        gtools.gazeto3d(gt)
                    )

                    name = [names[k]]
                    gaze = [str(u) for u in gaze]
                    gt = [str(u) for u in gt]
                    log = name + [",".join(gaze)] + [",".join(gt)]
                    outfile.write(" ".join(log) + "\n")

                    
                    # if count <= 10:
                    #     img_path = names[k]
                    #     img = cv2.imread(img_path)

                    #     if img is None:
                    #         print(f"Warning: Unable to read image {img_path}")
                    #         continue

                       
                    #     h, w, _ = img.shape
                    #     center = (w // 2, h // 2)

                    
                    #     gaze_end = (int(center[0] + float(gaze[0]) * w), int(center[1] - float(gaze[1]) * h))
                    #     gt_end = (int(center[0] + float(gt[0]) * w), int(center[1] - float(gt[1]) * h))

                    
                    #     img = draw_arrow(img, center, gaze_end, (0, 255, 0))  
                    #     img = draw_arrow(img, center, gt_end, (0, 0, 255))  
                    
                    #     save_path = os.path.join(logpath, f"fold_{fold+1}_epoch_{epoch}_image_{count}.png")
                    #     cv2.imwrite(save_path, img)

            loger = f"[Fold {fold+1} Epoch {epoch}] Total Num: {count}, avg: {accs/count}"
            outfile.write(loger)
            print(loger)
        outfile.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Testing')

    parser.add_argument('-s', '--source', type=str, help='config path about training')
    parser.add_argument('-t', '--target', type=str, help='config path about test')
    parser.add_argument('-f', '--fold', type=int, default=1, help='The fold number to test')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))
    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    print("=======================>(Begin) Config of training<======================")
    print(ctools.DictDumps(train_conf))
    print("=======================>(End) Config of training<======================")
    print("")
    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(test_conf))
    print("=======================>(End) Config for test<======================")

    main(train_conf.train, test_conf.test, args.fold)