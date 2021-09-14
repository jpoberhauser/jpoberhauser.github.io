# Working with aws neuron and Inferentia Instances


## Common problems and fixes

### 1. Getting "context deadline exceeded"

This I have found to mean that there is an auto-update that happens on the instance that causes the neuron-sdk installation to fail. 

Follow these instructions to re-install and attempt to run `neuron-top` again. 

https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-runtime/nrt-troubleshoot.html#neuron-driver-installation-fails


https://forums.aws.amazon.com/thread.jspa?threadID=337999&tstart=0

This has solved my inferentia/neuron issues a couple of times. 

If installation log is not available, check whether the module is loaded.

`$ lsmod | grep neuron`

If the above has no output then that means aws-neuron-dkms installation is failed.

Uninstall aws-neuron-dkms `sudo apt remove aws-neuron-dkms` or `sudo yum remove aws-neuron-dkms`

Install kernel headers for the current kernel `sudo apt install -y linux-headers-$(uname -r)` or `sudo yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)`

Install aws-neuron-dkms `sudo apt install aws-neuron-dkms` or `sudo yum install aws-neuron-dkms`

Restart runtime using `sudo systemctl restart neuron-rtd` command.



https://github.com/aws/aws-neuron-sdk/issues/325







