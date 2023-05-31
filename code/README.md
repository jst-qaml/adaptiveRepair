# AdRep

### Running experiments

The script AdaptiveExperiment.py runs a set of Adaptive Repairs.

```sh
$ python AdaptiveExperiment.py --help
usage: AdaptiveRunner [-h] [-g GPU] [-b BEGIN] [-e END] [-s SWITCH] [-w WINDOW] [-t THRESHOLD] [-p POSITIVES]

Runs an adaptive repair experiment

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU
  -b BEGIN, --begin BEGIN
  -e END, --end END
  -s SWITCH, --switch SWITCH
  -w WINDOW, --window WINDOW
  -t THRESHOLD, --threshold THRESHOLD
  -p POSITIVES, --positives POSITIVES

```

The arguments have the following meaning:
1. ```-g, --gpu``` the number of the gpu to run the optimization
2. ```-b, --begin``` ID of the first optimization
3. ```-e, --end``` ID of the last optimization
4. ```-s, --switch``` criterion for switching (stagnation or localisation)
5. ```-w, --window``` window hyperparameter
6. ```-t, --threshold``` threshold hyperparameter
7. ```-p, --positives``` number of positive inputs to sample

The default CNN that we repair is EnetB7. You can modify the script by replacing string "EnetB7" with "VGG16".

The script runs optimization for all the classes that we attempted to repair (see variable "classes" in the script).
