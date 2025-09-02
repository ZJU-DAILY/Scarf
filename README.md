# Scarf: Self-Adaptive Tuning via Multi-Objective Reinforcement Learning for Apache Flink

Scarf is an automatic configuration tuning framework for Apache Flink. It consists of:

- Knob selection acceleration through workload clustering
- Multi-objective Reinforcement Learning (MORL)-based offline-online learning
- Knowledge transfer via topology-agnostic GNN-based actor-critic network



## Build and Run

### Requirements

This tuner is implemented with Python 3.12. To run the tuner, install packages in `requirements.txt`.

This tuner is tested against Flink 2.0 running on Java 17 running YARN application mode with Hadoop 3.4.1.

The workloads are located in the `flink-jobs/` directory. You need to compile the JAR file and upload it to HDFS using `flink-jobs/build.sh`.

### The Tuning Pipeline

First, fill in the cluster address, job information and hyperparameters in `config/config.yaml`. The meaning of each configuration is described in `utils/config.py`. 

#### Knob Selection

##### From Scratch

Run:

```bash
python main.py --mode selection --stage coldstart --config config/config.yaml
```

An output folder will be created under `tuner.saveDir` in the config file.

##### Analyze Results

Place the output directory in `tuner.loadDir` in the config file, and run:

```bash
python main.py --mode selection --stage analysis --config config/config.yaml
```


##### Speed Up with History

Place the output directories of historical tasks in `selection/speedup.py`, and run:

```bash
python main.py --mode selection --stage cluster --config config/config.yaml
```


#### Offline Training

##### From Scratch

Remove the value of `tuner.loadDir` in the config file, fill in the selected knobs in the `knobs` section of the config file, and run:

```bash
python main.py --mode offline --config config/config.yaml
```

##### With Transfer

Place the output directory of the task to transfer from in `tuner.loadDir`, and run:

```bash
python main.py --mode offline --config config/config.yaml
```

#### Online Tuning

Place the output directory of the offline trained task in `tuner.loadDir`, and run:

```bash
python main.py --mode online --config config/config.yaml
```
