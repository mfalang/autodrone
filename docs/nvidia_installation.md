# Problems with NVIDIA drivers

## Installing NVIDIA drivers

1. Find out which version of the driver you should use by going to [this](https://www.nvidia.com/Download/index.aspx) link and entering the type of graphics card on the machine.
2. Then remove all NVIDIA drivers currently installed on the system

        sudo apt purge nvidia* libnvidia*
        apt list --installed | grep nvidia  // to verify that all is removed

3. Install correct drivers

        sudo add-apt-repository ppa:graphics-drivers
        sudo apt update
        sudo apt install nvidia-driver-470

4. Reboot machine
5. Verify that information about the graphics card appears when running. If not, see troubleshooting below.

        nvidia-smi

## Installing CUDA

TODO once I get the CUDA to work with darknet.

## Throubleshooting

### NVIDIA-SMI error

I had a lot of prolems getting the NVIDIA drivers and CUDA to work on system. The main issue was that there seemed to be something wrong with the drivers, as when I ran `nvidia-smi` I got an error saying "NVIDIA-SMI hasfailed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed an running.". The main issue seemed to be that there was some difference the kernel and the NVIDIA driver which was fixed by reinstalling the drivers. To fix the issue, I reinstalled the kernel headers for this specific kernel version, which solved the problem so that `nvidia-smi` now shows the correct output.

To reinstall the kernel headers, to the following:
1. Find out the kernel version

        cat /proc/version
        -> Linux version 5.4.0-89-generic ...

2. Reinstall the headers, where everything after "headers" is the same as found above

        sudo apt install --reinstall linux-headers-5.4.0-89-generic
