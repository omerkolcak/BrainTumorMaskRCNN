import mrcnn.utils
import mrcnn.config
import mrcnn.model

import glob
import skimage


class BTInferenceConfig(mrcnn.config.Config):
    NAME = "brain-tumor"

    GPU_COUNT = 1
    
    # 1 image for inference
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 1 + 1 # Tumor + Background 


inference_conf = BTInferenceConfig()

model = mrcnn.model.MaskRCNN(mode='inference', 
                             model_dir='./', 
                             config=inference_conf)

model.load_weights(filepath = 'models/mask_rcnn_brain-tumor_0022.h5', by_name = True)


def write_results(bb,scores,file_path):
    
    file = "DetectionResults/" + file_path
    
    with open(file,"w") as f:

        for box,score  in zip(bb,scores):
            line = "tumor " + str(score) + " "
            line += (str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]))
            line += "\n"
            
            f.write(line)   


train_images = glob.glob("TRAIN/*.jpg")
val_images = glob.glob("VAL/*.jpg")
test_images = glob.glob("TEST/*.jpg")


for image_path in train_images:
    
    image = skimage.io.imread(image_path)
    r = model.detect([image],verbose=1)[0]
    
    path = image_path.split("\\")[1].split(".")[0]
    
    file = "Training/" + path + ".txt"
    
    write_results(r['rois'],r['scores'],file)
    
for image_path in val_images:
    
    image = skimage.io.imread(image_path)
    r = model.detect([image],verbose=1)[0]
    
    path = image_path.split("\\")[1].split(".")[0]
    
    file = "Val/" + path + ".txt"
    
    write_results(r['rois'],r['scores'],file)
    
for image_path in test_images:
    
    image = skimage.io.imread(image_path)
    r = model.detect([image],verbose=1)[0]
    
    path = image_path.split("\\")[1].split(".")[0]
    
    file = "Test/" + path + ".txt"
    
    write_results(r['rois'],r['scores'],file)
    


        













