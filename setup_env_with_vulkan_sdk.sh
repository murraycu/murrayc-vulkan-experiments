# The downloaded vulkan-sdk contains pkg-config .pc files with incorrect prefix paths
# (not the same as wherever you downloaded the sdk).
#
# You need to correct these like so:
# (Note that the .pc files should contain full paths, not paths beginning with ~/.)
#
# cd ~/Downloads/vulkan-sdk-1.2/x86_64/lib/pkgconfig
# sed -i 's/\/root\/sdk-build\/1.2.131.2/$HOME\/Downloads\/vulkan-sdk-1.2/g' *.pc

export VULKAN_SDK_PATH=~/Downloads/vulkan-sdk-1.2/x86_64
export PKG_CONFIG_PATH=${VULKAN_SDK_PATH}/lib/pkgconfig
export LD_LIBRARY_PATH=${VULKAN_SDK_PATH}/lib
