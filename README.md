## Five bar robot
Five-bar planar parallel robots for pick and place operations.

## Developers information
### Install pre-commit
For install [pre-commit](https://pre-commit.com/#pre-commit-configyaml---hooks)

```bash
pip install pre-commit
```
Then after clone this repository and only once.

``` bash
pre-commit install
```

### Install
```bash
cd src/motion_planning
pip install -e .[dev]
```
### Create package
```bash
cd src/motion_planning
python setup.py bdist_wheel sdist
pip install -e .[dev]
```

### Documentation
#### Prerequisites
```bash
pip install sphinx
pip install sphinx-rtd-theme
pip install myst-parser
pip install sphinxcontrib-spelling
```
#### Generation
```bash
cd docs
make clean
make html
```

## Usage
See `src/motion_planning/examples`

## Install CoppeliaSim Api
In order to install CoppeliaSim Api follow this [steps](https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm)
Then, go to python's install directory `<path_to_python>/site-packages`. In that directory, create a file `name.pth`. Finally add the following path `<path_to_coppeliasim>/programming/zmqRemoteApi/clients/python` into the file.

## Notes
I suggest a project layout like:

```bash
five-bar-robot/
    doc/
        global-documentation
    src/
        package_1/
            CMakeLists.txt
            package.xml
        package_2/
            setup.py
            package.xml
            resource/package_2
        ...
        package_n/
            CMakeLists.txt
            package.xml
```
And a package layout similar to [this](https://docs.ros.org/en/galactic/Contributing/Developer-Guide.html#filesystem-layout) for our project.
