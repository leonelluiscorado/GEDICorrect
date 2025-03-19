#!/bin/bash -f

HOMDIR="$(pwd)/C-dependencies"

mkdir -p "$HOMDIR"

# Detect Linux distribution (WIP - Install dependencies based on distribution)
OS=$(lsb_release -is 2>/dev/null || source /etc/os-release && echo $ID)

# Install dependencies based on OS
echo "Detected OS: $OS"
if [[ "$OS" =~ (Ubuntu|Debian) ]]; then
    sudo apt update
    sudo apt install -y git make gcc g++ hdf5-tools libhdf5-dev libgsl-dev csh wget
else
    exit 1
fi

# Detect HDF5 installation path
HDF5_PATH=$(find /usr -name "libhdf5.so*" 2>/dev/null | head -n 1)

if [[ -n "$HDF5_PATH" ]]; then
    echo "HDF5 library found at: $HDF5_PATH"
else
    echo "Error: HDF5 library not found!" >&2
    exit 1
fi

# Set up environment variables
export ARCH=`uname -m`
export PATH=$PATH:./:$HOMDIR/bin/$ARCH:$HOMDIR/bin/csh
export GEDIRAT_ROOT=$HOMDIR/src/gedisimulator
export CMPFIT_ROOT=$HOMDIR/src/cmpfit-1.2
export GSL_ROOT=/usr/local/lib
export LIBCLIDAR_ROOT=$HOMDIR/src/libclidar
export HANCOCKTOOLS_ROOT=$HOMDIR/src/tools
export HDF5_LIB=$(dirname $HDF5_PATH)

# Persist environment variables
envFile="$HOME/.bashrc"
{
    echo "export ARCH=$(uname -m)"
    echo "export PATH=$PATH:./:$HOMDIR/bin/$ARCH:$HOMDIR/bin/csh"
    echo "export GEDIRAT_ROOT=$GEDIRAT_ROOT"
    echo "export CMPFIT_ROOT=$CMPFIT_ROOT"
    echo "export GSL_ROOT=$GSL_ROOT"
    echo "export LIBCLIDAR_ROOT=$LIBCLIDAR_ROOT"
    echo "export HANCOCKTOOLS_ROOT=$HANCOCKTOOLS_ROOT"
    echo "export HDF5_LIB=$HDF5_LIB"
} >> "$envFile"

source "$HOME/.bashrc"

# Set up directory structure
mkdir -p "$HOMDIR/src" "$HOMDIR/bin" "$HOMDIR/bin/$ARCH" "$HOMDIR/bin/csh"

pushd $HOMDIR/src
wget --show-progress https://www.physics.wisc.edu/~craigm/idl/down/cmpfit-1.2.tar.gz
tar -xvf cmpfit-1.2.tar.gz
popd

pushd $HOMDIR/src
git clone https://bitbucket.org/StevenHancock/libclidar
git clone https://bitbucket.org/StevenHancock/tools
git clone https://bitbucket.org/StevenHancock/gedisimulator

programList="gediRat gediMetric mapLidar collocateWaves lasPoints fitTXpulse"
cd $GEDIRAT_ROOT/
make clean

for program in $programList;do
  make THIS=$program
  make THIS=$program install
done

programList="gediRatList.csh listGediWaves.csh overlapLasFiles.csh filtForR.csh"
for program in $cshList;do
  cp $program $HOMDIR/bin/csh/
done

popd