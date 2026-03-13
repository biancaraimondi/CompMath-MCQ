# CompMath-MCQ
This repository contains the code and data for the paper ["The CompMath-MCQ Dataset: Are LLMs Ready for Higher-Level Math?"](https://arxiv.org/abs/2603.03334). The dataset consists of multiple-choice questions (MCQs) designed to evaluate the mathematical capabilities of large language models (LLMs).

### Prerequisites

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Use lm_eval library
The dataset is stored in `my_eval_task` folder as `mcq_lm_eval_data.jsonl`. This must be replaced in the .venv folder of lm_eval under `.venv/lib/python3.10/site-packages/lm_eval/tasks/my_custom_task/` to be used for testing. If in that path there is already a file named `mcq_lm_eval_data.jsonl`, replace it. If there is no such folder `my_custom_task`, create it. The folder must contain also the file `my_mcq_task.yaml`, which will define the task for lm_eval. An example of such file is provided in the `my_eval_task` folder.

#### To run the experiments using lm_eval:
Use the script `test_script.sh` provided. Make sure to modify the path to the models you want to test. The script will run the evaluation and save the results in a directory named `results/{model_name}`.

## Citation
If you find this dataset useful in your research, please consider citing our paper:

```bibtex
@article{raimondi2026compmath,
  title={The CompMath-MCQ Dataset: Are LLMs Ready for Higher-Level Math?},
  author={Raimondi, Bianca and Pivi, Francesco and Evangelista, Davide and Gabbrielli, Maurizio},
  journal={arXiv preprint arXiv:2603.03334},
  year={2026}
}
```