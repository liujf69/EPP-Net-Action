import os
import torch
import networks
import argparse
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.simple_extractor_dataset import SimpleFolderDataset

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    parser = argparse.ArgumentParser(description="Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='./checkpoints/final.pth', help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='./inputs', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='./outputs', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()

def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def main():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    sample_list = os.listdir(args.input_dir)
    sample_list.sort()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for sidx, sample_name in enumerate(tqdm(sample_list, desc='Parsing')):
        sample_dir = os.path.join(args.input_dir, sample_name)
        dataset = SimpleFolderDataset(root=sample_dir, input_size=input_size, transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
        # palette = get_palette(num_classes)
        logits_result_list = []
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                image, meta = batch
                img_name = meta['name'][0]
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].numpy()[0]
                h = meta['height'].numpy()[0]

                # print(image.shape) # torch.Size([32, 3, 473, 473])
                output = model(image.cuda())

                # save 119*119*1
                output = output[0][-1] # torch.Size([B, 20, 119, 119])

                logits_result = output.permute(0,2,3,1) # torch.Size([B, 119, 119, 20])
                logits_result_list.append(logits_result)
        
        sample_ts = torch.cat(logits_result_list, dim=0) # torch.Size([T, 119, 119, 20])

        sample_npy = sample_ts.data.cpu().numpy()
        parsing_result = np.argmax(sample_npy, axis=3) # [T, 119, 119]
        parsing_result = parsing_result[:, :, :, np.newaxis] # (T, 119, 119, 1)
        uint8_parsing_result = np.uint8(parsing_result)
        
        result_path = os.path.join(args.output_dir, sample_name + '.npy')
        np.save(result_path, uint8_parsing_result)

    return

if __name__ == '__main__':
    main()
