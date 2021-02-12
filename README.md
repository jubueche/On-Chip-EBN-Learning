# On-Chip Learning of EBNs (Efficient Balanced Networks)

This repository contains the scripts that were used to test the discrete learning rule on a network of 20 leaky integrate-and-fire neurons to drive
a random recurrent weight into a balanced regime, as well as a pdf showing the derivations of the learning rule in more detail. Link to the paper: https://arxiv.org/abs/2010.14353

## Requirements
Please install [rockpool](https://github.com/jubueche/Rockpool) using `$ pip install -e . --user` (more detailed instructions are in the repository).
Use `$ git fetch -a` to get all the branches and use `$ git checkout --track origin/paper/figure1` to checkout the correct branch.
This is a framework developed by [SynSense](synsense.ai) for training spiking neural networks with multiple backends.

Further requirements include `matplotlib,numpy` and `scipy`.

## Discrete Learning Rule
To run the script that was used in our paper, simply execute `$ python rockpool_discrete_learning.py`. This will train a randomly initialized recurrent network for 50 iterations and save the data necessary for plotting in a dictionary called `data.json`.

The figure can then be generated using `$ python plotting.py`.

<center>
<img src=figure1.png width="500">
</center>

If you are interested in the implementation of the learning rule, checkout `path/to/rockpool/rockpool/networks/gpl/net_deneve.py`.

## Simulations of the IC (Integrated Circuit)
The point of having a discrete learning rule for EBNs is that they make a hardware implementation much easier. We developed an IC and simulated it in two basic experiments that are also described in the paper. The data obtained from these simulation is located in the folder `Simulations/`. To generate the rest of the plots in the paper, execute `$ python plotting1.py` or `$ python plotting2.py`.

<center>
<img src=sim1_plot.png width="500">
</center>

<center>
<img src=sim2_plot.png width="500">
</center>


## Acknowledgement
```
@misc{büchel2020implementing,
      title={Implementing efficient balanced networks with mixed-signal spike-based learning circuits}, 
      author={Julian Büchel and Jonathan Kakon and Michel Perez and Giacomo Indiveri},
      year={2020},
      eprint={2010.14353},
      archivePrefix={arXiv},
      primaryClass={cs.ET}}
```
