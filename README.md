# On-Chip Learning of EBNs (Efficient Balanced Networks)

This repository contains the scripts that were used to test the discrete learning rule on a network of 20 leaky integrate-and-fire neurons to drive
a random recurrent weight into a balanced regime.

## Requirements
Please install [rockpool](https://github.com/jubueche/Rockpool) using `$ pip install -e . --user` (more detailed instructions are in the repository).
This is a framework developed by [SynSense](synsense.ai) for training spiking neural networks with multiple backends.

Further requirements include `matplotlib,numpy` and `scipy`.

## Discrete Learning Rule
To run the script that was used in our paper, simply execute `$ python rockpool_discrete_learning.py`. This will train a randomly initialized recurrent network for 50 iterations and save the data necessary for plotting in a dictionary called `data.json`.

The figure can then be generated using `$ python plotting.py`.

## Simulations of the IC (Integrated Circuit)
The point of having a discrete learning rule for EBNs is that they make a hardware implementation much easier. We developed an IC and simulated it in two basic experiments that are also described in the paper. The data obtained from these simulation is located in the folder `Simulations/`. To generate the rest of the plots in the paper, execute `$ python plotting1.py` or `$ python plotting2.py`.

## Acknowledgement
...missing