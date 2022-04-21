## Five bar robot

Five-bar planar parallel robots for pick and place operations.
## Install pre-commit
For install [pre-commit](https://pre-commit.com/#pre-commit-configyaml---hooks)

```bash
pip install pre-commit
```
Then after clone this repository

``` bash
pre-commit install
```
## Install
```bash
cd src/motion_planning
pip install -e .[dev]
```
## Usage
See `src/motion_planning/examples`

## Create package
```bash
cd src/motion_planning
python setup.py bdist_wheel sdist
pip install -e .[dev]
```

### Notes
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

### Authors
* Gonzalo Fernandez
* Jeremías Pino
