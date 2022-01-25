import glob
import train

import mrcnn.utils

training = glob.glob("TRAIN/annotations/bounding-boxes/*.txt")
test = glob.glob("TEST/annotations/bounding-boxes/*.txt")
val = glob.glob("VAL/annotations/bounding-boxes/*.txt")

train_dataset = train.BrainTumorDataset()        
train_dataset.load_dataset(dataset_dir = './', is_train = True)
train_dataset.prepare()

for i in range(len(training)):
    mask,_ = train_dataset.load_mask(i)
    
    boxes = mrcnn.utils.extract_bboxes(mask)

    
    file_name = training[i].split("\\")[1]
    
    if len(boxes) == 0:
        with open("../../Object-Detection-Metrics/BRAIN-TUMOR-DETECTION/TrainingGroundtruths/" + file_name, "a") as file:
            pass
    else:
        for box in boxes:
            with open("../../Object-Detection-Metrics/BRAIN-TUMOR-DETECTION/TrainingGroundtruths/" + file_name, "a") as file:
                line = ("tumor " + str(box[1]) + " " + str(box[0]) + " " + str(box[3]) + " " + str(box[2]) + "\n")
                
                file.write(line)
                
                
validation_dataset = train.BrainTumorDataset()        
validation_dataset.load_dataset(dataset_dir = './', is_train = False)
validation_dataset.prepare()


for i in range(len(val)):
    mask,_ = validation_dataset.load_mask(i)
    
    boxes = mrcnn.utils.extract_bboxes(mask)
    
    file_name = val[i].split("\\")[1]
    
    if len(boxes) == 0:
        with open("../../Object-Detection-Metrics/BRAIN-TUMOR-DETECTION/ValidationGroundtruths/" + file_name, "a") as file:
            pass
    else:
        for box in boxes:
            with open("../../Object-Detection-Metrics/BRAIN-TUMOR-DETECTION/ValidationGroundtruths/" + file_name, "a") as file:
                line = ("tumor " + str(box[1]) + " " + str(box[0]) + " " + str(box[3]) + " " + str(box[2]) + "\n")
                
                file.write(line)


                
                
                
            
        
    
    
    
    
            
    
        
        
            
