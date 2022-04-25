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

## Installation

Download the source code and move working directory to clone repository:

```bash
git clone https://github.com/isce-framework/PLAnT-ISCE3.git
cd PLAnT-ISCE3
```

Install PROTEUS via conda/setup.py (recommended):

```bash
python setup.py install
```

Or via pip:

```bash
pip install .
```

Or via environment path setup:

```bash
export PLANT_ISCE3_HOME=$PWD
export PATH=${PATH}:${PLANT_ISCE3_HOME}/bin
```
