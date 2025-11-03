![](https://github.com/leonelluiscorado/GEDICorrect/blob/main/readme/GEDICorrectLOGO.png)<br/>

![GitHub Release](https://img.shields.io/github/v/release/leonelluiscorado/GEDICorrect)
![GitHub License](https://img.shields.io/github/license/leonelluiscorado/GEDICorrect)
![GitHub Repo stars](https://img.shields.io/github/stars/leonelluiscorado/GEDICorrect)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/leonelluiscorado/GEDICorrect/total)

## GEDICorrect: A Python Framework for GEDI Geolocation Correction at the Orbit, Beam or Footprint-Level Using Multiple Criteria and Parallel Processing Methods

An open-source Python framework for precise GEDI geolocation correction using small-footprint ALS data, designed with simplicity and accessibility in mind. GEDICorrect integrates multiple methods, criteria, and metrics, including waveform matching, terrain matching, and relative height (RH) profile matching, to achieve refined geolocation accuracy at the orbit, beam, or footprint levels. By leveraging advanced similarity metrics - such as Pearson and Spearman waveform correlations, Curve Root Sum Squared Differential Area (CRSSDA), and Kullback-Leibler divergence - GEDICorrect ensures precise alignment between GEDI measurements and simulated data.

Additionally, GEDICorrect incorporates parallel processing strategies using Python’s multiprocessing capabilities, enabling efficient handling of large-scale GEDI and ALS datasets. This scalability makes the framework practical for global-scale applications while maintaining accuracy and computational efficiency. This framework works as an extension to ![GEDI Simulator](https://bitbucket.org/StevenHancock/gedisimulator/src/master/), developed by Steven Hancock and the GEDI Science Team, which we are thankful for their valuable contributions. More specifically, GEDICorrect uses the _gediRat_ and _gediMetrics_ programs from GEDI Simulator to simulate GEDI footprints using the ALS data, and extract metrics from those footprints.

By addressing critical barriers in geolocation correction with an open-source, user-friendly design, this framework enables a better assessment of canopy structure that can be applied to a wide range of fields, from advancing our understanding of carbon sequestration to supporting more informed planning and conservation efforts.

## Installation

**Requirements**:
- GCC: 11.4 or later (to install ![GEDI Simulator](https://bitbucket.org/StevenHancock/gedisimulator/src/master/))
- Python: 3.11 or later
- Anaconda: 23.11.0 or 24.4.0 or later
- OS: Linux or Windows WSL (Ubuntu)

**Installation Steps**:

Always keep a copy of the most recent update of GEDICorrect, to ensure it functionality.

_________________________________

For Windows Users:
- GEDICorrect Support for Windows (natively) is currently unsupported, however, it provides Windows Subsystem for Linux (WSL) in the most recent versions of Windows (10 or 11). Before installing GEDICorrect, it is recommended that you install the ![Ubuntu WSL](https://learn.microsoft.com/en-us/windows/wsl/install). After following the instructions, start your Ubuntu subsystem and follow the subsequent steps inside the virtual machine.

_________________________________

To install GEDICorrect:

1. Clone the repository. Navigate to the cloned repository's directory.
2. Execute the Bash script `install_hancock_tools.bash` located in the root directory of the repository. This script will install GEDI Simulator and all the necessary dependencies to perform GEDI waveform simulations. To execute it, first give it executable permissions with `chmod +x install_hancock_tools.bash` and then execute it with `./install_hancock_tools`.
3. After installing GEDI Simulator, setup the Anaconda virtual environment (must have ![Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) installed first) for this repository by installing its required dependencies using the provided `environment.yml` with `conda env create -f environment.yml`.
4. Activate the virtual environment with `conda activate GEDICorrect`. You're set up!

## Getting Started

**Data Requirements**:

![](https://github.com/leonelluiscorado/GEDICorrect/blob/main/readme/AlignL1BL2A.drawio.png)<br/>

GEDICorrect requires two set of data to correct GEDI footprints: ALS flight and intersecting GEDI orbits on a Study Area. ALS must be provided in `.las` files. For GEDI, only L1B and L2A data products are currently supported, and GEDICorrect only accepts **merged** L1B-L2A data products. A merge script (recommended) is available at `utils/align_l1b_l2a.py`. Please follow the instructions provided with:
`python3 align_l1b_l2a.py --help`.

**Running GEDICorrect**:

To execute the framework, simply follow the instructions provided in the `gedi_correct.py` script by running this command: `python3 gedi_correct.py --help`. The user must always specify:
- _.las_ files directory
- **Merged** GEDI files directory
- An output directory where to save the corrected files.

**Examples**:

Switch the file directories provided in the examples with your desired directories, inside quotation marks ("").

```
python3 gedi_correct.py --las_dir "/home/leonel/abrantes_studyarea" --granules_dir "/home/leonel/gedi_merged_abrantes" --out_dir "/home/leonel/abrantes_correct"
```
Executes GEDICorrect with default footprint correction settings: at the orbit-level using Pearson's Correlation metric.
______________
```
python3 gedi_correct.py --las_dir "/home/leonel/abrantes_studyarea" --granules_dir "/home/leonel/gedi_merged_abrantes" --out_dir "/home/leonel/abrantes_correct" --mode "beam" --criteria "wave_pearson kl"
```
Executes GEDICorrect on a study area at the Beam-level, using the Pearson Correlation and KL metrics on the waveforms to calculate the similarities.
______________
```
python3 gedi_correct.py --las_dir "/home/leonel/abrantes_studyarea" --granules_dir "/home/leonel/gedi_merged_abrantes" --out_dir "/home/leonel/abrantes_correct" --mode "footprint" --time_window 0.04 --criteria "rh_distance" --parallel --n_processes 8
```
Executes GEDICorrect at the Footprint-level, using the CRSSDA on the RH profile similarity metric with a time window of 0.04. This command also lets GEDICorrect run in parallel, using 8 processes to process GEDI data.
______________

For more information on commands and options, please use
```
python3 gedi_correct.py --help
```

## Example Dataset

An example dataset is available on ![Zenodo](https://zenodo.org/records/17494713). It contains a merged L1B/L2A GEDI orbit for a small area in Portugal, and the accompanying ALS point cloud dataset. We encourage you to follow the dataset instructions before test running GEDICorrect. Before running this test, the user must convert the ".laz" files to ".las" using the `utils/convert_las.py` script located in the repository. Follow the instructions of the script for further information. Make sure you have atleast 200GB of space available for the ".las" dataset.

## Contributing to this repository

This project is in its early stages so any contributions are welcome with a well documented/explained issue and implementation! If you encounter any problems with GEDICorrect or at the Installation/Execution steps, please open an issue on this repository with the steps described to replicate the problem.

## References

Dubayah, R., Blair, J.B., Goetz, S., Fatoyinbo, L., Hansen, M., Healey, S., Hofton, M., Hurtt, G., Kellner, J., Luthcke, S., & Armston, J. (2020) The Global Ecosystem Dynamics Investigation: High-resolution laser ranging of the Earth’s forests and topography. Science of Remote Sensing, p.100002. https://doi.org/10.1016/j.srs.2020.100002

Hancock, S., Armston, J., Hofton, M., Sun, X., Tang, H., Duncanson, L.I., Kellner, J.R. and Dubayah, R., 2019. The GEDI simulator: A large-footprint waveform lidar simulator for calibration and validation of spaceborne missions. Earth and Space Science. https://doi.org/10.1029/2018EA000506

- GEDI L1B Geolocated Waveform Data Global Footprint Level - [GEDI01_B](https://lpdaac.usgs.gov/products/gedi01_bv001/)
- GEDI L2A Elevation and Height Metrics Data Global Footprint Level - [GEDI02_A](https://lpdaac.usgs.gov/products/gedi02_av002/)

Check out our other repository ![GEDI-Pipeline](https://github.com/leonelluiscorado/GEDI-Pipeline), an unified workflow to download GEDI data!

## Citing this project

Corado, L., Godinho, S. (2025) GEDICorrect: A Python Framework for GEDI Geolocation Correction at the Orbit, Beam or Footprint-Level Using Multiple Criteria and Parallel Processing Methods. version 0.3.0, available at: https://github.com/leonelluiscorado/GEDICorrect

### Funding

This work was conducted within the framework of the GEDI4SMOS project (Combining LiDAR, radar, and multispectral data to characterize the three-dimensional structure of vegetation and produce land cover maps), financially supported by the Directorate-General for Territory (DGT) with funds from the Recovery and Resilience Plan (Investimento RE-C08-i02: Cadastro da Propriedade Rústica e Sistema de Monitorização da Ocupação do Solo).


