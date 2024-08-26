# OSCER HPC Tutorial

This repository serves as an introduction to the OSCER system and a template for using it to train large models that require multiple gpus and multiple compute nodes.  After reading through the contents of the scripts the hope is you have gained some understanding of usage and best practices for the OSCER system, as well as its purpose and function.

## Table of Contents
- [Installation](#installation)
- [Execution Walkthrough](#execution-walkthrough)
- [Roadmap](#roadmap)
- [License](#license)

## Installation

To install the package just clone this repository into your home directory.  You can reach your home directory with the `cd ` command.

## Execution Walkthrough

### Imports

This project requires the use of a large number of packages, some of which are implemented in python, and others of which are compiled for this machine.  Some of those packages in turn rely on system modules that we are not permitted to install on an HPC system, but which must be installed by system administrators.  Luckily, this system has many users and the packages you want are already likely installed, but OSCER is a distributed system not all modules are installed on all machines within the system.  You can load the module you would like with the `module load <module name>` command.  This process is streamlined for python packages and you can use the conda or mamba modules to manage virtual environments that handle the CUDA and python modules you will need for deep learning.

### The SBatch Script

The Simple Linux Utility for Resource Management (SLURM) is the resource manager for the OSCER system, and many other systems like it.  The resource manager manages a set of partitions (which are mathematically subsets) of the resources within the system.  Each partition is assigned nodes each of which has a name.  You can use the `sinfo` command to get a list of the nodes and their status.  There are many helpful resource which describe the different functionalities of slurm such as [this one](https://wiki.umiacs.umd.edu/umiacs/index.php/SLURM), so that discussion is better left to those resources.  I will just address one resource request below.

Each request is given as a set of command line arguments in a bash script that are defined on lines beginning with `SBATCH`.  With the exception of installing packages, monitoring resources, and reading text files <span style="color: red;">**YOU SHOULD NEVER RUN ANY CODE DIRECTLY**</span> it should all be submitted as a resource request with a bash script.  When you ssh into the system all subsequent actions you take are on one of a very few nodes which all users must share and which have very few resources.  It is best practice not to use them for data transfers such as `rsync` and `scp` instead use the `dtn2.oscer.ou.edu` node.  An example of such a request follows.

```bash
#!/bin/bash
# the name of the partition you are submitting to
#SBATCH --partition=gpu
# the number of nodes you will use, usually only 1 is required.  If your code is not designed to use more then more will not be better.
#SBATCH --nodes=2
# the number of processes you will launch.  This should be equal to the number of nodes
#SBATCH --ntasks=2
# Thread count, or the number of hypercores you plan to use.  This is not enforced.
#SBATCH --cpus-per-task=32
# The number of gpus you require each node to have
#SBATCH --gres=gpu:2
# memory (RAM) you will use on each machine.  Provide an upper bound, this is not enforced
#SBATCH --mem=64G
# Where you would like your stdout and stderr to appear
#SBATCH --output=/home/you/OUAI_Demo/out.txt
#SBATCH --error=/home/you/OUAI_Demo/err.txt
# The maximum time your job can take (most partitions limit this)
#SBATCH --time=24:00:00
# job name which will appear in queue
#SBATCH --job-name=demo
# if you fill this out slurm will email you job status updates, consider sending them to a folder.
#SBATCH --mail-user=your_email@ou.edu
#SBATCH --mail-type=ALL
# the working directory for your job.  This must exist.
#SBATCH --chdir=/home/you/OUAI_Demo/
#################################################

# this is how we get the ip address of the node that will be used for the master process
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")

echo Node IP: $head_node_ip

# using Dr. Fagg's conda setup script
. /home/fagg/tf_setup.sh
# activating a version of my environment
conda activate /home/jroth/.conda/envs/mct
# logging in to weights and biases
wandb login your_api_key

# launching a run that will be executed over multiple compute nodes
srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$head_node_ip:64425" \
main.py

```

### The Training Script

When the python script is run, this will execute.

```python
if __name__ == '__main__':
    # get the number of processes we will be using in parallel
    # there will be one for each gpu we are using
    world_size = int(os.environ["WORLD_SIZE"])
    # the 'rank' of the current process, a unique id for the current gpu
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method('spawn')
    # then we launch the main method
    main()
```

The main method is what handles the setup and cleanup of the process group which we use to orchestrate communication across the ranks of the job.  These ranks must communicate to keep the weights of the neural network we are training in sync during the job to ensure correct computations are performed.  Most of this magic will be handled for us in this example, but it is possible to utilize the pytorch NCCL bindings to manually execute such communication.

```python
def main():
    # parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    # set up process group
    setup()
    # do the experiment
    training_process(args)
    # clean up process group
    cleanup()
```

I keep all of the model training code in this `training_process` function.  This is the main part of the script that will orchestrate dataset construction, model weight updates, and state saving.

```python
def training_process(args):

    rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    # getting the local index of the device for this process
    device = torch.device(rank % torch.cuda.device_count())
    # many torch functionalities will require that the torch.cuda.current_device() is set.
    torch.cuda.set_device(device)
    # if this is the master process then we will start logging to weights and biases
    if rank == 0:
       wandb.init(project='OUAI Demo',
                   entity='ai2es',
                   name='Tuning DINOv2',
        )
    ...
```

Next, we will need to build the model object that we are training.  Please see the entire script for more detailed description of the model that is used here.  First we will be making use of a pre-trained backbone for vision called DINOv2, and then we will be fine-tuning it.

```python
    ...
    DINOv2 = dino_model()

    # append a linear layer to the frozen backbone to utilize "transfer learning"
    # the model must be sent .to(device) prior to wrapping
    model = LinearProbe(DINOv2).to(device)

    # model wrapper argument for activation checkpointing which saves memory

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000)

    # wrap the model with the Fully-Sharded Data Parallel wrapper which handles many things for us

    model = FSDP(model, 
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True)
            )

    # define the optimizer, this must occur after wrapping

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ...

```

We must also use some sort of dataset when training our models.  There are two key objects in pytorch which manage data, the first is a `Dataset` the second is a `DataLoader`.  The first can really be any `Iterable` which supports `__len__` and `yield`s elements of the appropriate shape.  The second is a multithreaded batch creation object that aggregates the samples into batches and `yield`s them in an order for training.  Note that in python `**` is the dictionary unpacking operator.

```python
    ...
    # instance preprocessing necessary for using the pre-trained backbone
    DINOv2_transform = dino_transforms()
    # dataset objects for the CIFAR100 image classification task
    train_ds = torchvision.datasets.CIFAR100('./cifar100', train=True, transform=DINOv2_transform, download=True)
    val_ds = torchvision.datasets.CIFAR100('./cifar100', train=False, transform=DINOv2_transform, download=True)
    # distributed sampler that makes sure each rank gets different instances from the dataset in each epoch
    sampler_train = DistributedSampler(train_ds, rank=rank, num_replicas=world_size, shuffle=True)
    sampler_val = DistributedSampler(val_ds, rank=rank, num_replicas=world_size, shuffle=False)

    cuda_kwargs = {'num_workers': 12, 
                    'pin_memory': True, 
                    'shuffle': False}

    loader_kwargs = {'batch_size': args.batch_size, 
                     'sampler': sampler_train, 
                     'multiprocessing_context': 'forkserver', 
                     'persistent_workers': False,
                     'drop_last': False}
    
    loader_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_ds, **loader_kwargs)

    loader_kwargs = {'batch_size': args.batch_size, 
                     'sampler': sampler_val, 
                     'multiprocessing_context': 'forkserver', 
                     'persistent_workers': False,
                     'drop_last': False}
    
    loader_kwargs.update(cuda_kwargs)

    val_loader = DataLoader(val_ds, **loader_kwargs)
    ...
```

Now all that is left to do is actually train the model.  First we will train the linear layer.

```python
    ...
    print(os.environ['RANK'], 'linear probe')
    for epoch in tqdm(range(args.epochs)):
        # to update the randomness used for shuffling in the sampler
        sampler_train.set_epoch(epoch)
        train(model, opt, train_loader)
        gc.collect()
        vacc = torch.tensor(test(model, val_loader, silent=True)).to(device)
        dist.all_reduce(vacc, op=dist.ReduceOp.AVG)
        if rank == 0:
            wandb.log({'validation accuracy': vacc})
            print({'validation accuracy': vacc})
    ...
```

Finally, we fine-tune the backbone which was frozen.

```python
    ...
    states = model.state_dict()

    DINOv2 = dino_model()

    model = LinearProbe(DINOv2).to(device)

    model.load_state_dict(states)
    # the Full Parameter Fine-Tuning Object.
    model = FPFT(model)

    model = FSDP(model, 
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True)
            )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr / 8.0)

    print(os.environ['RANK'], 'finetuning')
    for epoch in tqdm(range(args.epochs)):
        sampler_train.set_epoch(epoch)
        train(model, opt, train_loader)
        gc.collect()
        vacc = torch.tensor(test(model, val_loader, silent=True)).to(device)
        dist.all_reduce(vacc, op=dist.ReduceOp.AVG)
        if rank == 0:
            wandb.log({'validation accuracy': vacc})
            print({'validation accuracy': vacc})
```


## License
Copyright 2024 Jay Rothenberger (jay.c.rothenberger@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.