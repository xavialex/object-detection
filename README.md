# Preventing dangerous situations

In our everyday lives, lots of potentially harmful situations can be found, especially in industrial jobs where dangerous objects/products are treated. To help address this problem, this project is meant to detect abnormal dispositions of this objects as soon as possible to prevent future damage. For this purpose, an object detection model is applied to detect the instances of the classes of interest in an image, for example in a video stream, and make calculations to discern if the object is straight or laid, and reach an emergency alert in consequence.

## Dependencies

Running the executable provided in *app/* avoids worrying about dependencies. See the **Use** section below for more information.

It's also possible to run the source script in Python. If you're a conda user, you can create an environment from the ```danger_prevention_env.yml``` file using the Terminal or an Anaconda Prompt for the following steps:

1. Create the environment from the ```danger_prevention_env.yml``` file:

    ```conda env create -f danger_prevention_env.yml```
2. Activate the new environment:
    > * Windows: ```activate danger_prevention```
    > * macOS and Linux: ```source activate danger_prevention``` 

3. Verify that the new environment was installed correctly:

    ```conda list```
    
You can also clone the environment through the environment manager of Anaconda Navigator.

## Use

### Executable

The simplest way to use the program is to run the executable located in the *app/* folder. A default configuration will be applied, where a console and a floating window with the processed video stream of the main camera attached to the computer will show up. Furthermore, you'll be able to see a GUI accessing to your *localhost:8082*. A JSON message is send to *localhost:8082/alerts* containing:
> * **class:** The class of the object that's been detected.  
> * **msg:** An informative message about the potentially hazard position of the object.
> * **confidence:** The confidence with which the model detected that object.

Lastly, it's possible to alter the configuration variables from the *properties.txt* file to modify the behavior of the program. This are the modifiable entries:
> * **VIDEO_SOURCE:** Integer from 0 to N, where 0 is the main camera connected to your computer and so on. Default 0.
> * **UI_PORT:** Port used by the script to send the GUI. Default 8082.
> * **MODEL:** Name of the TensorFlow object detection model that's going to be used to process the video stream. They are located in *TF Object Detection Models/trained_models/*. The default and only model available is the *ssdlite_mobilenet_v2_coco_2018_05_09*. To search more models depending on your necessities/system capabilities go to [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and download them into *TF Object Detection Models/trained_models/* before specifying its name in *properties.txt*.
> * **DISCRIMINATIVE_CLASSES:** Comma separated integers form 1 to 90 which corresponds with the class id's of the classes in COCO dataset, the image dataset used to train these models. To see a list of what classes are available, go to *object_detection/data/mscoco_label_map.pbtxt* and put the ones of your interest in *properties.txt*.

After saving the *properties.txt* modifications, they'll be applied the next time you run the executable. The result is a new window containing the original image but with the detection bounding boxes in it. If any of the objects detected are laid they're detected as in a dangerous position. To stop the program, select the floating window and press *'q'* or '*Ctrl+C'* in the console.

### Source

After having available the *danger_prevention* environment, launch *harm_detection.py* to see the processed input video signal from a wired cam. It's also possible to modify the *properties.txt* as explained above. The result is the same as in the executable version. Press the *'q'* button in the resulting floating window or *Ctrl+C* in the console to close the program.
