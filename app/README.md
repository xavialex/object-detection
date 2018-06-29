# People detection

This executable performs person detection within an input video stream from a wired camera. 

## Use

Simply run the executable to see the people detector in action. It's also possible to alter the configuration variables from the *app.properties* file to modify the behavior of the program. This are the modifiable entries:
> * **VIDEO_SOURCE:** Integer from 0 to N, where 0 is the main camera connected to your computer and so on. Default 0.
> * **UI_PORT:** Port used by the script to send the GUI. Default 8082.
> * **MODEL:** Name of the TensorFlow object detection model that's going to be used to process the video stream. They are located in *TF Object Detection Models/trained_models/*. The default and only model available is the *ssdlite_mobilenet_v2_coco_2018_05_09*. To search more models depending on your necessities/system capabilities go to [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and download them into *TF Object Detection Models/trained_models/* before specifying its name in *properties.txt*.
> * **DISCRIMINATIVE_CLASSES:** Comma separated integers form 1 to 90 which corresponds with the class id's of the classes in COCO dataset, the image dataset used to train these models. To see a list of what classes are available, go to *object_detection/data/mscoco_label_map.pbtxt* and put the ones of your interest in *app.properties*.

After saving the *app.properties* modifications, they'll be applied the next time you run the executable. The result is a new window containing the original image but with the detection bounding boxes in it. To stop the program, select the floating window and press *'q'* or '*Ctrl+C'* in the console.
