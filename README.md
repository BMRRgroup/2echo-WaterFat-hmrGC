# Dual-echo water-fat separation with hierarchical multi-resolution graph-cuts

This repository is an extension of the [fieldmapping-hmrGC](https://github.com/BMRRgroup/fieldmapping-hmrGC) repository to implement a water-fat separation algorithm with two acquired echos of flexible echo times.

The algorithm builds up on the following two publications:
- *Jonathan K. Stelter, Christof Boehm, Stefan Ruschke, Kilian Weiss, Maximilian N. Diefenbach, Mingming Wu, Tabea Borde, Georg P. Schmidt, Marcus R. Makowski, Eva M. Fallenberg and Dimitrios C. Karampinos; Hierarchical multi-resolution graph-cuts for water–fat–silicone separation in breast MRI, IEEE Transactions on Medical Imaging, DOI: 10.1109/TMI.2022.3180302, https://ieeexplore.ieee.org/document/9788478*
- *Holger Eggers, Bernhard Brendel, Adri Duijndam and Gwenael Herigault; Dual-echo Dixon imaging with flexible choice of echo times, Magnetic Resonance in Medicine, DOI: 10.1002/mrm.22578, https://doi.org/10.1002/mrm.22578*

The method was applied and described in the context of water-specific T1 and T2 mapping. Examples can be found in the respective [repository](https://github.com/BMRRgroup/liver-t1t2-mapping):

*Jonathan Stelter, Kilian Weiss, Lisa Steinhelfer, Veronika Spieker, Elizabeth Huaroc Moquillaza, Weitong Zhang, Marcus R. Makowski, Julia A. Schnabel, Bernhard Kainz, Rickmer F. Braren and Dimitrios C. Karampinos; Simultaneous whole‐liver water T1 and T2 mapping with isotropic resolution during free‐breathing, NMR in Biomedicine, DOI: 10.1002/nbm.5216, https://doi.org/10.1002/nbm.5216*

## Setup

### Requirements

Development was performed using python-3.8. Requirements are stated in `setup.py`:
* numba (v0.55.0 recommended)
* pymaxflow (v1.2.13 recommended)
* opencv-python (v4.5.5.64 recommended)
* scipy (v1.7.3 recommended)
* *for GPU matrix computations:* cupy (v9.5.0 recommended)

Unit tests are stored in `/tests`. Pytest and h5py are needed as addtional requirements to run the tests.

### Installing

The package can be easily installed using Pip:

Direct installation from GitHub:
```
pip install git+https://github.com/BMRRgroup/2echo-WaterFat-hmrGC
```

or clone the repository to use the developement mode (recommended):
```
git clone https://github.com/BMRRgroup/2echo-WaterFat-hmrGC
pip install -e 2echo-WaterFat-hmrGC
```

### Quick start example
```
from hmrGC_dualEcho.dual_echo import DualEcho

# Input arrays and parameters
signal = ...   # complex array with dim (nx, ny, nz, nte)
mask = ...   # boolean array with dim (nx, ny, nz)
params = {}
params['TE_s'] = ...   # float array with dim (nte)
params['centerFreq_Hz'] = ...   # float (in Hz, not MHz)
params['fieldStrength_T'] = ...   # float
params['voxelSize_mm'] = ...   # recon voxel size with dim (3)
params['FatModel'] = {}
params['FatModel']['freqs_ppm'] = ...   # chemical shift difference between fat and water peak, float array with dim (nfatpeaks)
params['FatModel']['relAmps'] = ...   # relative amplitudes for each fat peak, float array with dim (nfatpeaks)

# Initialize DualEcho object
g = DualEcho(signal, mask, params)

# Perform graph-cut method
g.perform()   # methods with different parameters can be defined using the dual_echo.json file
 
# Access separation results
phasormap = g.phasormap
waterimg = g.images['water']
fatimg = g.images['fat']
```

## Authors and acknowledgment
**Main contributors:**
* Jonathan Stelter - [Body Magnetic Resonance Research Group, TUM](http://bmrr.de)
* Christof Boehm - [Body Magnetic Resonance Research Group, TUM](http://bmrr.de)

## License
This project builds up on the [PyMaxflow](https://github.com/pmneila/PyMaxflow) library and the [Maxflow C++ implementation](https://pub.ist.ac.at/~vnk/software.html) by Yuri Boykov and Vladimir Kolmogorov and is therefore released under the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
