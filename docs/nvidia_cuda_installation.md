# Installing NVIDIA drivers and CUDA

## Installing NVIDIA drivers

1. Find out which version of the driver you should use by going to 
[this](https://www.nvidia.com/Download/index.aspx) link and entering the type 
of graphics card on the machine.
2. Then remove all NVIDIA drivers currently installed on the system

        sudo apt purge nvidia* libnvidia*
        apt list --installed | grep nvidia  // to verify that all is removed

3. Install correct drivers

        sudo add-apt-repository ppa:graphics-drivers
        sudo apt update
        sudo apt install nvidia-driver-470

4. Reboot machine
5. Verify that information about the graphics card appears when running. If 
not, see troubleshooting below.

        nvidia-smi

## Installing CUDA

The following steps are based on 
[this tutorial](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130),
but as there seems to be some issues with CUDA 10.1 and Darknet, we will install
CUDA 10.2 instead.

1. Remove previous installations

        sudo rm /etc/apt/sources.list.d/cuda*
        sudo apt remove --autoremove nvidia-cuda-toolkit

2. Set up CUDA ppa

        sudo apt update
        sudo add-apt-repository ppa:graphics-drivers
        
        sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

        sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

        sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

3. Install CUDA 10.2

        sudo apt update
        sudo apt install cuda-10-2
        sudo apt install libcudnn7

4. Add CUDA 10.2 to path by adding the following lines to `~/.profile`

        # Set PATH for cuda 10.2 installation
        if [ -d "/usr/local/cuda-10.2/bin/" ]; then
                export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
                export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
        fi

5. Reboot machine and verify that versions are correct

        nvcc -V
        -> Should be 10.2

        /sbin/ldconfig -N -v $(sed ‘s/:/ /’ <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
        -> Should be "libcudnn.so.7 -> libcudnn.so.7.6.5"

## Compile Darknet

Perform the following steps before compiling Darknet for the first time.

1. Find out the compute capability of the graphics card by finding it on
[the CUDA wiki page](https://en.wikipedia.org/wiki/CUDA).
2. Add the following line to the file `darknet_ros/darknet_ros/CMakeLists.txt`, 
(where the number is the compute capability found above)

        -gencode arch=compute_75,code=sm_75

3. Build in release mode (if it does not build, see troubleshooting below)

        catkin_make -DCMAKE_BUILD_TYPE=Release


## Throubleshooting

### NVIDIA-SMI error

I had a lot of prolems getting the NVIDIA drivers and CUDA to work on system. 
The main issue was that there seemed to be something wrong with the drivers, as 
when I ran `nvidia-smi` I got an error saying "NVIDIA-SMI hasfailed because it 
couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA 
driver is installed an running.". The main issue seemed to be that there was 
some difference the kernel and the NVIDIA driver which was fixed by reinstalling 
the drivers. To fix the issue, I reinstalled the kernel headers for this 
specific kernel version, which solved the problem so that `nvidia-smi` now shows 
the correct output.

To reinstall the kernel headers, to the following:
1. Find out the kernel version

        cat /proc/version
        -> Linux version 5.4.0-89-generic ...

2. Reinstall the headers, where everything after "headers" is the same as found 
above

        sudo apt install --reinstall linux-headers-5.4.0-89-generic

### GCC version error in build

If the build fails because it requires GCC to be no higher than version 6, then
install GCC 6 and compile using that instead of the default once.

```
sudo apt install gcc-6
catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-6
```