import json

import mrcnn.model


def json_to_txt(folder_name):
    f = open(folder_name)
    
    sub_folders = folder_name.split("/")
    main_folder = sub_folders[0] + "/" + sub_folders[1] + "/" + "annotations" + "/"
    
    annotations = json.load(f)
    
    for key in annotations:
        
        img_name = annotations[key]['filename']
        
        for region in annotations[key]['regions']:
            
            attributes = region['shape_attributes']
            
            
            file_name_bb = img_name.split(".")[0] + ".txt"
            file_name_mask = img_name.split(".")[0] + ".txt"        
            
            file_bb = open(main_folder + "bounding-boxes/" + file_name_bb, "a")
            file_mask = open(main_folder +  "masks/" + file_name_mask, "a")
            
            if "all_points_x" in attributes:
            
                x = attributes['all_points_x']
                y = attributes['all_points_y']
                
                # Bounding Boxes
                xmin = min(x)
                ymin = min(y)
                xmax = max(x)
                ymax = max(y)
                
    
                file_bb.write(str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n")
                
                
                for point in x:    
                    file_mask.write(str(point) + " ")
                        
                file_mask.write("\n")
                    
                for point in y:    
                    file_mask.write(str(point) + " ")
                
                file_mask.write("\n")

            
    
json_to_txt("brain-tumor-detection/TEST/annotations_test.json")
json_to_txt("brain-tumor-detection/TRAIN/annotations_train.json")
json_to_txt("brain-tumor-detection/VAL/annotations_val.json")



