This folder contains all necessary files to run the custom MCMC algorithm designed by the Craig Allen and Dr. Edward Stites.

A requirements.txt file is provided with necessary Python packages to be installed, with versions. 
It is reccomended an Anaconda environment is used to install and control packages, but pip can be used instead by running >>> pip install -r path/to/requirements.txt

The RAS_MCMC_tutorial.ipynb Python notebook walks one through the basic usage of the MCMC_tools.py file. For use on the Yale HPC cluster, MCMC Job Tutorial is provided with detailed instructions on using main.py.

The MCMC_tools file was designed for general usage with any rules defined in rule_functions.py with a standard format, with the RAS model as a default. It is up to the user to define their own ODE models and rule functions if a different model is to be used.

For any questions, please contact craig.allen@yale.edu or craig.fraser.allen@gmail.com
- Craig Allen, 8-2-2024, 