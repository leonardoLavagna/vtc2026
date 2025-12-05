# vtc2026 

Implementations for the conference [VTC2026](https://events.vtsociety.org/vtc2026-spring/) about "A Quantum Method for Constrained Vehicle Dynamics and Green-Wave Optimization".


## What's in here?
Here you can find
- `config` which contains configuration files such as `gate_options.py` and `paths.py` needed to specify the hyperparameters of the quantum machine learning algorithm and its training options.
- `core` which contains the core modules of the quantum machine learning algorithm, from the `data_loader.py` to the `training.py` utilities.
- `run_pipeline.py` is the file to execute in order to train the quantum machine learning algorithm and to select the best resultin model with `best_instance.py`.
- `examples.ipynb` which contains some step-by-step carried out examples of the model (see the companion article for more details).
- `data.csv` contains a classical simulation data baseline yielded by a traditional [MPC](https://en.wikipedia.org/wiki/Model_predictive_control).
- `real_data.csv` contains real measurements of an instrumented bus used as an experimental benchmark (see the companion article for more details).
- `results_3-20.csv` contains optimal results in the time interval $t\in [3,20]$ yielded by the best model obtained from `run_pipeline.py` and `best_instance.py` where the `core/data_loader.py` file containind the function `load_dataset` should be called with `load_dataset(t_min=3, t_max=20)`.
- `requirements.txt` contains the requirements to reproduce the experiments we carried out.
- `LICENCE` is the MIT licence.


## Use this repository
If you want to use the code in this repository in your projects, please cite explicitely our work, and
* Clone the repository with `git clone https://github.com/leonardoLavagna/vtc2026`
* Install the requirements with `pip install -r requirements.txt`


## Contributing
We welcome contributions to enhance the functionality and performance of the models. Please submit pull requests or open issues for any improvements or bug fixes.

## License
This project is licensed under the MIT License.


## Citation
Cite this repository or one of the associated papers, such as:

```
...
```
