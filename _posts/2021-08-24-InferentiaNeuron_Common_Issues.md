# Working with aws neuron and Inferentia Instances


## Common problems and fixes

### 1. Getting "context deadline exceeded"

This I have found to mean that there is an auto-update that happens on the instance that causes the neuron-sdk installation to fail. 

Follow these instructions to re-install and attempt to run `neuron-top` again. 

https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-runtime/nrt-troubleshoot.html#neuron-driver-installation-fails

This has solved my inferentia/neuron issues a couple of times. 

### 2. Getting random boxes and predictions

This one is a bit harder to debug, but this usually happens when you send an image for inference to the neuron-compiled model and the results are basically random boxes on the image. The solution to this is actually the same as in #1, given that you knew that you were getting predictions that made sense before, and all of a sudden they stop making sense. 

### 3. Converting models to neuron


