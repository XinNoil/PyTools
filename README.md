# PyTools
## Description
python tools

## Usage
### How to use PyTools anywhere?

You can add PyTools to your environment path by adding the following content to `.bashrc`:

```
export PYTHONPATH=/home/wjk/Workspace/PyTools:$PYTHONPATH
```

### How to import functions
For example
```python
from mtools import save_json
from datahub.db import DB
```

## Modules

### mtools

Tools including **io, numpy and built-in functions**.

Since we `import *` in `mtools.__init__.py`, you can import any function `func`  by :

```python
from mtools import func
```

#### mtools.io

| function                                              | Description                  |
| ----------------------------------------------------- | ---------------------------- |
| `save_json(filename, obj)`                            | save obj to a json file      |
| `load_json(filename)`                                 | load obj from a json file    |
| `save_h5(filename, obj)`                              | save obj to a h5 file        |
| `load_h5(filename)`                                   | load obj from h5 file        |
| `csvwrite(filename, data, delimiter=',', fmt='%.4f')` | write data to a csv file     |
| `csvread(filename)`                                   | read data from a csv file    |
| `save_mat(filename, **kwargs)`                        | save variables to a mat file |

> **save_json**

 - **Description**: save obj to a json file

 - **Parameters**:

`filename* (str)`: filename

`obj (object)`: any object want to save

 - **Usage**:

```python
save_json(filename, obj)
```

> **load_json**

 - **Description**: load obj from a json file

 - **Parameters**:

`filename* (str)`: filename

 - **Usage**:

```python
obj = load_json(filename)
```

> **save_h5**

save obj to a h5 file

> **load_h5**

load obj from h5 file

> **csvwrite**

 - **Description**: write data to a csv file

 - **Parameters**:

`filename* (str)`: filename

`data (numpy.ndarray)`: numpy array want to save

`delimiter`: default as `,`

`fmt`: default as four decimal places

 - **Usage**:

```python
csvwrite(filename, data)
csvwrite(filename, data, delimiter=' ')
csvwrite(filename, data, delimiter=' ', fmt='%.8f')
```

> **csvread**

 - **Description**: read data from a csv file

 - **Parameters**:

`filename* (str)`: filename

 - **Usage**:

```python
data = csvread(filename)
```

> **save_mat**

 - **Description**: save variables to a mat file

 - **Parameters**:

`filename* (str)`: filename

`**kwargs (dict)`: variable length parameters

 - **Usage**:

```python
save_mat(filename, train_data=train_data, test_data=test_data)
```

#### mtools.io.np

Extra tools for numpy.

#### mtools.io.py

Extra tools for python built-in functions or classes.

### mtools.datahub

#### mtools.datahub.db

#### mtools.datahub.wifi

#### mtools.datahub.plots