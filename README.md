## Required package and software

- anaconda3.0
- tensorflow-gpu==1.14.0
- tensorflow-datasets==1.2.0
- tensorflow-gan==1.0.0.dev0
- tensorflow-probability==0.7.0

see the export environment configuration.

## Install

1. prepare the virtual environment:

`conda create --name myenv anaconda python=3.6`

2. install tensorflow and related packages:

```pip install tensorflow-gpu==1.14.0```

```pip install tensorflow-datasets==1.2.0```

```pip install tensorflow-gan==1.0.0.dev0```

```pip install tensorflow-probability==0.7.0```

3. Open the jupyter notebook and switch the kernel to the created virtual environment. For how to add the created virtual environment to kernel, please refer to following commands:

``` source activate myenv ```

``` python -m ipykernel install --user --name myenv --display-name "Python (myenv)" ```

4. Run the code in jupyter notebooks.