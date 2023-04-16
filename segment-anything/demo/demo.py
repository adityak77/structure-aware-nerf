import argparse
import matplotlib.pyplot as plt
import numpy as np

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def main(img):
    model_type = 'vit_h'
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamAutomaticMaskGenerator(sam)

    masks = predictor.generate(img)

    plt.figure(figsize=(20,20))
    plt.imshow(img)
    show_anns(masks)
    plt.axis('off')
    plt.savefig('demo/output.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='demo/desk.jpg')
    args = parser.parse_args()

    img = plt.imread(args.img)
    main(img)