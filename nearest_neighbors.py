import os
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from pprint import pprint
from torchvision.transforms import *
from utils import check_dir
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg
from tqdm import tqdm
os.environ["TORCH_HOME"] = "/project/dl2022s/panneer/cache"
os.environ["MPLCONFIGDIR"] = "/project/dl2022s/panneer/cache"
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-init', type=str,
                        default="")
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--data_folder', type=str, help="folder containing the data (crops)", default='/project/dl2022s/panneer/crops/images/')
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors",
                     args.weights_init.replace("/", "_").replace("models", "")))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # model
    #raise NotImplementedError("TODO: build model and load weights snapshot")
    model = ResNet18Backbone(pretrained=False).cuda()
    #pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.weights_init)
    pretrained = torch.load('/project/dl2022s/panneer/results/lr0.0005_bs48__local/models/ckpt_epoch1.pth')
    

    backbone = { key.replace("backbone.", "") : value
        for key, value in pretrained["teacher"].items() if not key.startswith("head")}
    model.load_state_dict(backbone)
    

    #model.load_state_dict(torch.load('/project/dl2022s/panneer/results/lr0.0005_bs48__local/models/ckpt_epoch1.pth', map_location=torch.device('cuda')))


    # for neares neighbors we don't need the last fc layer
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # dataset
    val_transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    #raise NotImplementedError("Load the validation dataset (crops), use the transform above.")
    val_data = DataReaderPlainImg(os.path.join(args.data_folder, str(args.size), "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=True)

    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    query_indices = [29]
    nns = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    display_image = []
    for idx, img in enumerate(val_loader):
        if idx not in query_indices:
            continue
        display_image.append(img)

        print("Computing NNs for sample {}".format(idx))
        img = img.to(device)
        closest_idx, closest_dist = find_nn(model, img, val_loader, 5)
        #raise NotImplementedError("TODO: retrieve the original NN images, save them and log the results.")
        print("Closest neighbors for image : ", idx, " -> ", str(closest_idx))
        print("Closest neighbors distance for image : ", idx, " -> ", str(closest_dist))

    for idx, img in enumerate(val_loader):
        if idx in closest_idx:
            display_image.append(img)

    fig = plt.figure(figsize=(32, 32))
    plt.title("comparing nearest neighbour for image 29 (1st image)")
    for i in range(1, len(display_image) + 1):
        img = display_image[i - 1]
        img = torch.squeeze(img).permute(1, 2, 0)
        fig.add_subplot(3, 3, i)
        plt.imshow(img)
    plt.savefig("nearest_neighbour_image_29")
    plt.show()

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
    #raise NotImplementedError("TODO: nearest neighbors retrieval")
    #query_output = model(query_img)['out']
    query_output = model(query_img)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t = tqdm(loader)
    distances = []
    for images in t:
        images = images.to(device)
        output = model(images)
        print(output.shape)
        d = torch.norm(query_output - output, p=2)
        print(d.shape, d)
        distances.append(d.item())

    distances = np.array(distances)
    closest_idx = np.argsort(distances)[:k]
    closest_dist = distances[closest_idx]
    return closest_idx, closest_dist


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args) 
