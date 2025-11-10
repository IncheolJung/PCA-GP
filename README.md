README
=========

- This is a brief description how to run the code.
- Before running the code, to setup Python environment, refer to dec/user-manual/main.pdf

- To run in sequential:
```
#!/bin/bash
python main.py
```

OR

```
#!/bin/bash
bash run_node.sh	# change gpytorch into your anaconda environment
```

- For parallel run:
```
#!/bin/bash
bash run_mpi.sh run_node.sh
```
