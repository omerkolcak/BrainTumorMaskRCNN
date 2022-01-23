from numpy import zeros, asarray, uint8, int32
import numpy as np

import mrcnn.utils
import mrcnn.config
import mrcnn.model

import glob
import skimage


CLASS_IDS = {1 : "tumor"}


class BrainTumorConfig(mrcnn.config.Config):
    NAME = "brain-tumor"

    GPU_COUNT = 1
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    
    NUM_CLASSES = 1 + 1 # Tumor + Background 

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000 

class BrainTumorDataset(mrcnn.utils.Dataset):
    
    def load_dataset(self, dataset_dir, is_train=True):
        
        self.add_class("brain-tumors", 1, "tumor")
        
        if is_train:
            bb_folder = dataset_dir + "/" + "TRAIN/" + "annotations/" + "bounding-boxes/"
            mask_folder = dataset_dir + "/" + "TRAIN/" + "annotations/" + "masks/"
            images_folder = dataset_dir + "/" + "TRAIN/"
        else:
            bb_folder = dataset_dir + "/" + "VAL/" + "annotations/" + "bounding-boxes/" 
            mask_folder = dataset_dir + "/" + "VAL/" + "annotations/" + "masks/"
            images_folder = dataset_dir + "/" + "VAL/"
        
        bounding_boxes = glob.glob(bb_folder + "*.txt")
        masks = glob.glob(mask_folder + "*.txt")
        images = glob.glob(images_folder + "*.jpg")
        
        image_id = 0
        for path in zip(images,bounding_boxes,masks):
            
            img = skimage.io.imread(path[0])
            height, width = img.shape[:2]
                        
            self.add_image("brain-tumors", image_id, path = path[0]
                           , bb_path = path[1], mask_path = path[2], 
                           width = width, height = height)
            image_id += 1
            
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        
        all_points_x = []
        all_points_y = []
        
        x = []
        y = []
        
        with open(mask_path, "r") as file:
            line = file.readline()
            
            x_or_y = 0
            while line:
                for p in line.split(" "):
                    if x_or_y % 2 == 0:
                        if p.isnumeric():
                            x.append(int(p))
                    else:
                        if p.isnumeric():
                            y.append(int(p))
                
                if x_or_y % 2 == 0:
                    all_points_x.append(x)
                    x = []
                else:
                    all_points_y.append(y)
                    y = []
                
                x_or_y += 1
                line = file.readline()
                        
        num_instances = int(x_or_y / 2)
        mask = np.zeros([info["height"], info["width"], num_instances],dtype=np.uint8)
        
        for i in range(num_instances):
            rr, cc = skimage.draw.polygon(all_points_y[i], all_points_x[i])
            mask[rr, cc, i] = 1
        
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    
    
train_dataset = BrainTumorDataset()        
train_dataset.load_dataset(dataset_dir = './brain-tumor-detection', is_train = True)
train_dataset.prepare()

validation_dataset = BrainTumorDataset()        
validation_dataset.load_dataset(dataset_dir = './brain-tumor-detection', is_train = False)
validation_dataset.prepare()
           
tumor_config = BrainTumorConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=tumor_config)

model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=tumor_config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')