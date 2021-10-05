# How to get Object Detection Android App using TFLite:


1.) Clone this repo to your local:

`https://github.com/hunglc007/tensorflow-yolov4-tflite`

2.) Make sure the changes on this PR are in the files you pulled:

`https://github.com/hunglc007/tensorflow-yolov4-tflite/pull/162/commits/5b11ef71eb5e7a700aa3fd2e8b9a66235a5ec118`

Notable changes here:

  * app/src/main/java/org/tensorflow/lite/examples/detection/tflite/YoloV4Classifier.java 
 
 
3.) Convert weights to `.tflite`

Have your `.tflite` and your `.txt` with classes ready

4.) Put both your `.tflite` and your `.tx`t with classes here:

`tensorflow-yolov4-tflite/android/app/src/main/assets`



* In that folder, you should see:

```
coco.txt               kite.jpg               own_classes.txt        yolov4-416-fp32.tflite
detect.tflite          labelmap.txt           yolov3tiny-416.tflite  yolov4tiny-416.tflite
```


5.) Go to edit the file (local) :

`tensorflow-yolov4-tflite/android/app/src/main/java/org/tensorflow/lite/examples/detection/MainActivity.java`

There, look for lines: (right now they are lines 77 & 79)

```
    private static final String TF_OD_API_MODEL_FILE = "<name_of_your_.tflite>";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/<name_of_classes.txt>";
```



6.) Next go to edit the file (local): 

`tensorflow-yolov4-tflite/android/app/src/main/java/org/tensorflow/lite/examples/detection/DetectorActivity.java`

look for lines: (55-60)

```
    private static final int TF_OD_API_INPUT_SIZE = 416;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "own_model.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/own_classes.txt";
```

Notice how you have to change the TF_OD_API_INPUT_SIZE to be 416,
Change TF_OD_API_IS_QUANTIZED to false
Change TF_OD_API_MODEL_FILE to reflect your .tflite model
Change TF_OD_API_LABELS_FILE to reflect your classes.txt file


7.) Build gradle

8.) Run app on device






## Helpful Links:

* https://medium.com/datadriveninvestor/how-to-train-your-own-custom-model-with-tensorflow-object-detection-api-and-deploy-it-into-android-aeacab7fa76f
