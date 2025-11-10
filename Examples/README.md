Instructions on the example files
==========

- Before running examples, please refer to the user manual to set up the anaconda environment

- All the examples assume that the environment name is "gpytorch." Change the environment name in each bash file if a different environment name is used.

- Run the chosen example using bash
```
#!/bin/bash
bash run example_circylinder.sh
```

- To run parallel version, use "run_mpi" template available in the main directory.
```
#!/bin/bash
bash ../run_mpi.sh example_circylinder.sh
```

- Below is the list of examples
	- sphere
	- rounded-cylinder
	- prime-aircraft
	- tanker-aircraft

- In case of error, try remove configuration files and the retry.
```
#!/bin/bash
cd .. && rm *.cfg && cd Examples
bash run example_circylinder.sh
```
