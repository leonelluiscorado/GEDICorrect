#!/usr/bin/env bash

set -euo pipefail

GEDICORRECT_PREFIX="${GEDICORRECT_PREFIX:-$(pwd)/C-dependencies}"
GEDICORRECT_SKIP_SYSTEM_PACKAGES="${GEDICORRECT_SKIP_SYSTEM_PACKAGES:-0}"
GEDICORRECT_PERSIST_ENV="${GEDICORRECT_PERSIST_ENV:-1}"
GEDICORRECT_BUILD_ALL_TOOLS="${GEDICORRECT_BUILD_ALL_TOOLS:-0}"
ARCH="$(uname -m)"
LIBCLIDAR_REF="${LIBCLIDAR_REF:-8535ed4428f71445f06c01981e6b9758a53555b3}"
HANCOCKTOOLS_REF="${HANCOCKTOOLS_REF:-d0fd539365c4efa61aab8dffadaf84e9f7e4c507}"
GEDISIMULATOR_REF="${GEDISIMULATOR_REF:-a204c6de6dcad1dfc175bfe8e956101ce576ced5}"
CMPFIT_REF="${CMPFIT_REF:-4f3449cff762ba7b259b02933fe582337e925081}"

if [[ "${GEDICORRECT_SKIP_SYSTEM_PACKAGES}" != "1" ]]; then
    if [[ -r /etc/os-release ]]; then
        source /etc/os-release
    fi

    case "${ID:-unknown}" in
        ubuntu|debian)
            if command -v sudo >/dev/null 2>&1; then
                SUDO=sudo
            elif [[ "$(id -u)" == "0" ]]; then
                SUDO=""
            else
                echo "sudo is required to install system dependencies." >&2
                exit 1
            fi
            ${SUDO} apt-get update
            ${SUDO} apt-get install -y \
                git make gcc g++ hdf5-tools libhdf5-dev libgsl-dev \
                libgdal-dev libgeotiff-dev libtiff-dev csh ca-certificates
            ;;
        *)
            echo "Automatic GEDI Simulator installation supports Debian and Ubuntu only." >&2
            echo "Use the GEDICorrect container on Windows and macOS." >&2
            exit 1
            ;;
    esac
fi

HDF5_LIBRARY="$(find /usr/lib /usr/local/lib -name 'libhdf5.so*' -print -quit 2>/dev/null || true)"
if [[ -z "${HDF5_LIBRARY}" ]]; then
    echo "HDF5 library not found. Install libhdf5-dev and retry." >&2
    exit 1
fi

export ARCH
export GEDIRAT_ROOT="${GEDICORRECT_PREFIX}/src/gedisimulator"
export CMPFIT_ROOT="${GEDICORRECT_PREFIX}/src/cmpfit-1.2"
export GSL_ROOT="/usr"
export LIBCLIDAR_ROOT="${GEDICORRECT_PREFIX}/src/libclidar"
export HANCOCKTOOLS_ROOT="${GEDICORRECT_PREFIX}/src/tools"
export HDF5_LIB="$(dirname "${HDF5_LIBRARY}")"
export LIBRARY_PATH="${HDF5_LIB}:${LIBRARY_PATH:-}"
export PATH="${GEDICORRECT_PREFIX}/bin:${GEDICORRECT_PREFIX}/bin/${ARCH}:${GEDICORRECT_PREFIX}/bin/csh:${PATH}"

mkdir -p "${GEDICORRECT_PREFIX}/src" "${GEDICORRECT_PREFIX}/bin/${ARCH}" "${GEDICORRECT_PREFIX}/bin/csh"

clone_if_missing() {
    local repository="$1"
    local destination="$2"
    local reference="$3"
    if [[ ! -d "${destination}/.git" ]]; then
        git clone --no-tags "${repository}" "${destination}"
    fi
    git -C "${destination}" checkout --detach "${reference}"
}

cmpfit_source="${GEDICORRECT_PREFIX}/src/mpyfit"
clone_if_missing https://github.com/evertrol/mpyfit.git "${cmpfit_source}" "${CMPFIT_REF}"
mkdir -p "${CMPFIT_ROOT}"
cp "${cmpfit_source}/mpyfit/cmpfit/mpfit-orig.c" "${CMPFIT_ROOT}/mpfit.c"
cp "${cmpfit_source}/mpyfit/cmpfit/mpfit.h" "${CMPFIT_ROOT}/mpfit.h"
cp "${cmpfit_source}/mpyfit/cmpfit/DISCLAIMER" "${CMPFIT_ROOT}/DISCLAIMER"

clone_if_missing https://bitbucket.org/StevenHancock/libclidar "${LIBCLIDAR_ROOT}" "${LIBCLIDAR_REF}"
clone_if_missing https://bitbucket.org/StevenHancock/tools "${HANCOCKTOOLS_ROOT}" "${HANCOCKTOOLS_REF}"
clone_if_missing https://bitbucket.org/StevenHancock/gedisimulator "${GEDIRAT_ROOT}" "${GEDISIMULATOR_REF}"

pushd "${GEDIRAT_ROOT}" >/dev/null
programs=(gediRat gediMetric)
if [[ "${GEDICORRECT_BUILD_ALL_TOOLS}" == "1" ]]; then
    programs+=(mapLidar collocateWaves lasPoints fitTXpulse)
fi
for program in "${programs[@]}"; do
    make clean THIS="${program}"
    make THIS="${program}"
    cp "${program}" "${GEDICORRECT_PREFIX}/bin/${ARCH}/${program}"
    ln -sfn "${GEDICORRECT_PREFIX}/bin/${ARCH}/${program}" "${GEDICORRECT_PREFIX}/bin/${program}"
done

csh_programs=(gediRatList.csh listGediWaves.csh overlapLasFiles.csh filtForR.csh)
for program in "${csh_programs[@]}"; do
    if [[ -f "${program}" ]]; then
        cp "${program}" "${GEDICORRECT_PREFIX}/bin/csh/"
    fi
done
popd >/dev/null

if [[ "${GEDICORRECT_PERSIST_ENV}" == "1" ]]; then
    environment_file="${HOME}/.bashrc"
    marker="# GEDICorrect GEDI Simulator"
    if ! grep -Fq "${marker}" "${environment_file}" 2>/dev/null; then
        {
            echo ""
            echo "${marker}"
            echo "export ARCH=${ARCH}"
            echo "export PATH=${GEDICORRECT_PREFIX}/bin:${GEDICORRECT_PREFIX}/bin/${ARCH}:${GEDICORRECT_PREFIX}/bin/csh:\$PATH"
            echo "export GEDIRAT_ROOT=${GEDIRAT_ROOT}"
            echo "export CMPFIT_ROOT=${CMPFIT_ROOT}"
            echo "export GSL_ROOT=${GSL_ROOT}"
            echo "export LIBCLIDAR_ROOT=${LIBCLIDAR_ROOT}"
            echo "export HANCOCKTOOLS_ROOT=${HANCOCKTOOLS_ROOT}"
            echo "export HDF5_LIB=${HDF5_LIB}"
        } >> "${environment_file}"
    fi
fi

echo "GEDI Simulator installed in ${GEDICORRECT_PREFIX}"
echo "Run 'gedicorrect check' in a new shell to verify the installation."
