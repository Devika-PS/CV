import os
import random
import argparse
import torch
from pprint import pprint
from torchvision.transforms import *
from utils import check_dir
from models.pretraining_backbone import ResNet18Backbone, DINOHead
from data.pretraining import DataReaderPlainImg
from pretrain import MultiCropWrapper

import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help="folder ^^containing the data (crops)", default="./datasets/crops/images/256")
    parser.add_argument('--weights-init', type=str, default="results/pretrain/lr0.0005_bs48__local/models/ckpt_epoch2.pth")
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
   
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors",
                     args.weights_init.replace("/", "_").replace("models", "").replace(".pth", "")))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # model
    #raise NotImplementedError("TODO: build model and load weights snapshot")

    file_name = "./results/lr0.0005_bs48__local/models/ckpt_epoch9.pth" 
    
    state = torch.load(args.weights_init)
    teacher_state_dict = state["teacher"]
    # backbone_state_dict = {key.replace("backbone.", "") : value 
    #                     for key, value in teacher_state_dict.items() if not key.startswith("head")}

    teacher = ResNet18Backbone(pretrained=False)
    teacher = MultiCropWrapper(teacher, DINOHead(
        512, 128, norm_last_layer=False,
    ))
    # teacher.load_state_dict(backbone_state_dict)
    teacher.load_state_dict(teacher_state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    teacher.to(device)

    # dataset
    val_transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    #raise NotImplementedError("Load the validation dataset (crops), use the transform above.")
    val_data = DataReaderPlainImg(os.path.join(args.data_folder, "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0,
                                             pin_memory=True, drop_last=True)


    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    query_indices = [0, 64]
    nns = {}

    for query_id, img in enumerate(val_loader):
        if query_id not in query_indices:
            continue
        print("Computing NNs for sample {}".format(query_id))
        neighs_idx, neighs_dist = find_nn(teacher, img, val_loader, 5)
        nns[query_id] = {
            "idx" : neighs_idx,
            "dist" : neighs_dist
        }
    
    report(nns, val_data, args)

def report(nns, val_data, args):

    for query_id, result in nns.items():
        neighs_idx = result["idx"]
        neighs_dist = result["dist"]

        fig = plt.figure()
        ax = fig.add_subplot(1, 6, 1)
        imgplot = plt.imshow(val_data[query_id].squeeze(0).permute(1,2,0))
        ax.set_title('Query')

        for n, id in enumerate(neighs_idx):

            ax = fig.add_subplot(1, 6, n + 2)
            imgplot = plt.imshow(val_data[id].squeeze(0).permute(1,2,0))
            ax.set_title(f'd={neighs_dist[n]:.2f}')

        plt.savefig(f"{args.output_folder}/query_{query_id}.png")

def find_nn(model, query_img, loader, k):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
         model: the model for computing the features
         query_img: the image of which to find the NNs
         loader: the loader for the dataset in which to look for the NNs
         k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    # raise NotImplementedError("TODO: nearest neighbors retrieval")
    distances = torch.zeros(len(loader))

    model.eval()
    with torch.no_grad():
        query_feature_repr = model(query_img.cuda())
        print(query_feature_repr.shape)

        for id, sample in enumerate(loader):
            sample_feature_repr = model(sample.cuda())
            
            distances[id] = l2_distance(sample_feature_repr, query_feature_repr)
        
        sort_ids = distances.argsort()
        closest_idx = sort_ids[1:k+1]
        closest_dist = distances[closest_idx]

    return closest_idx, closest_dist

def l2_distance(x, y):
    z = (x - y).view(-1)
    return torch.sqrt(z.dot(z))

if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args) 
