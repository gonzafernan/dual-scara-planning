from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='motion_planning',
      version='0.0.x',
      description='Five bar robot motion planning package.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      py_modules=["FiveBar", "Path", "TimeLaw", "utilities"],
      package_dir={'': 'motion_planning'},
      install_requires=["numpy >= 1.2", "matplotlib >= 3.5", "pyserial >= 3.5"],
      extras_require={
          "dev": [
              "pytest >= 6.0", "check-manifest", "twine", "flake8 >= 3.9",
              "pre-commit >= 2.17"
          ]
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"
      ],
      url="https://github.com/gonzafernan/five-bar-robot",
      author="Jeremías Pino Demichelis",
      author_email="jerepinos@gmail.com")
