# PLAnT-ISCE3
PLAnT-ISCE3: Polarimetric Interferometric Lab and Analysis Tool (PLAnT) scripts for the InSAR Scientific Computing Environment 3 (ISCE3)

---

PLAnT-ISCE3 is a general-purpose toolbox that uses the open-source Polarimetric Interferometric Lab and Analysis Tool (PLAnT) framework to provide an “easy-to-use” command-line interface (CLI) for the open-source InSAR Scientific Computing Environment 3 (ISCE3) framework and leverage ISCE3 capabilities. PLAnT-ISCE3 delivers an interface to ISCE3 modules/functionalities focusing on the end-user. Additionally, since most ISCE3 modules can only be accessed externally via ISCE3 C++ or Python application programming interfaces (APIs), i.e., not through ISCE3 command-line interfaces (CLI), PLAnT-ISCE3 provides unique access to many ISCE3 functionalities that are not directly exposed to the end-user.

ISCE3 repository: https://github.com/isce-framework/isce3

PLAnT repository: https://gitlab.com/plant/plant

---

Copyright 2022, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

---

# Installation

## 1. From conda-forge (Recommended)

```bash
conda install plant-isce3 -c conda-forge
```

## 2. Conda/pip
1. Clone PLAnT-ISCE3 repository
2. Install PLAnT-ISCE3 using pip:
```bash
git clone https://github.com/isce-framework/plant-isce3.git
cd plant-isce3
python -m pip install .
```

## 3. From a conda environment, manually setting PATH and PYTHONPATH environment variables
1. Clone PLAnT-ISCE3 repository
2. Install PLAnT-ISCE3 dependencies listed in `requirements.txt`
3. Add PLAnT-ISCE3 parent folder to PYTHONPATH variable
4. Add PLAnT-ISCE3 applications folder to PATH variable

```bash
git clone https://github.com/isce-framework/plant-isce3.git
cd plant-isce3
export PLANT_ISCE3_HOME=$PWD/src
export PYTHON_PATH=${PYTHON_PATH}:${PLANT_ISCE3_HOME}
export PATH=${PATH}:${PLANT_ISCE3_HOME}/plant_isce3
```
