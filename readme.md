# Comparison of [OpenAI GPT-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) and [DeepSeek V3](https://api-docs.deepseek.com/news/news1226) models for code bug fixing

## Project structure:
* [prompts](prompts) - prompt templates 
* [utils](source/utils) - utility functions

## How models are compared
* [SWE-Bench](https://www.swebench.com/lite.html) dataset with issues related to [SymPy](https://github.com/sympy/sympy) repository is used to test models' bug fixing capabilities
* Each issue in the dataset includes creation commit and accepted solution git patch to compare with


### How each SWE Bench SymPy issue is processed
> Main logic is in [tester.py](source/utils/tester.py)
* Perform git checkout to pre-issue commit
* Use retriever to get up to 100 potentially useful files to solve the issue
    * Prompt available [here](prompts/retriever.txt)
* Execute solver algorithm
    * Take problem description and files from the retriever step as input
    * At each step the solver can either output a git patch or request more files for its context (up to 3 times)
    * Prompt available [here](prompts/solver.txt)
* Test solutions
    * Both OpenAI and DeepSeek are compared against the correct patch from the SWE Bench dataset
    * Testing performed by both models to eliminate bias
    * Prompt available [here](prompts/patch_checker.txt)