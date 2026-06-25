# PLAnT-ISCE3
PLAnT-ISCE3: Polarimetric Interferometric Lab and Analysis Tool (PLAnT) scripts for the InSAR Scientific Computing Environment 3 (ISCE3)

---

PLAnT-ISCE3 is a general-purpose toolbox that uses the open-source Polarimetric Interferometric Lab and Analysis Tool (PLAnT) framework to provide an “easy-to-use” command-line interface (CLI) for the open-source InSAR Scientific Computing Environment 3 (ISCE3) framework and leverage ISCE3 capabilities. PLAnT-ISCE3 delivers an interface to ISCE3 modules/functionalities focusing on the end-user. Additionally, since most ISCE3 modules can only be accessed externally via ISCE3 C++ or Python application programming interfaces (APIs), i.e., not through ISCE3 command-line interfaces (CLI), PLAnT-ISCE3 provides unique access to many ISCE3 functionalities that are not directly exposed to the end-user.

ISCE3 repository: https://github.com/isce-framework/isce3

PLAnT repository: https://gitlab.com/plant/plant

# License
Licensing and legal notices are provided in LICENSE.txt.

# Installation

**Requirements:**
- Python >= 3.9
- ISCE3 >= 0.25.0
- PLAnT >= 0.8.4
- Dependencies listed in `requirements.txt`

## 1. Install from conda-forge (Recommended)

Install PLAnT-ISCE3 and all its dependencies from `conda-forge`:

```bash
conda install plant-isce3 -c conda-forge
```

## 2. Install from Source Using pip

1. Clone the PLAnT-ISCE3 repository:
```bash
git clone https://github.com/isce-framework/plant-isce3.git
cd plant-isce3
```

2. Install dependencies:
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate plant_isce3_env

# Or install dependencies manually
conda install --yes --file requirements.txt
```

3. Install PLAnT-ISCE3:
```bash
python -m pip install .
```

## 3. Manual Installation with Environment Variables

For development or advanced users who want to run PLAnT-ISCE3 without installing:

1. Clone PLAnT-ISCE3 repository:
```bash
git clone https://github.com/isce-framework/plant-isce3.git
cd plant-isce3
```

2. Install dependencies:
```bash
conda install --yes --file requirements.txt
```

3. Set environment variables:
```bash
# Add PLAnT-ISCE3 to PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:$PWD/src

# Add PLAnT-ISCE3 applications to system PATH
export PATH=${PATH}:$PWD/src/plant_isce3/
```

**Note:** These environment variables need to be set each time you start a new shell session. To make them persistent, add them to your ~/.bashrc, ~/.zshrc, or equivalent shell startup file.
