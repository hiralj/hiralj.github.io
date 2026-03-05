### Motivation
It is recommended to run Jupyter from it's own Virtual Environment (instead of system's) to avoid cluttering and any dependency issues.
We assume Mac system here.

### Steps
1. Install Jupyter Lab.
    - `which jupyter` from terminal should be able to find it and give system path. May be like "/usr/local/bin/jupyter".
2. Create Python Virtual Environment from within your project folder: `python3 -m venv venv`
    - The second "venv" here is the name of the folder to create. Can be anything.
3. Activate the venv: `source venv/bin/activate`
4. Install IPython Kernel, needed for Jupyter: `pip install ipykernel`
5. Add environment as Jupyter Kernel: `python -m ipykernel install --user --name=<your_env_name> --display-name="<Your Display Name>"
`
    - e.g.: "python -m ipykernel install --user --name=venv --display-name="AI Cohort"
6. Start Jupyter Lab: `jupyter lab`
7. You should see new Kernel (which is dedicated env) on creating new notebook.
<img width="1057" height="525" alt="Screenshot 2026-03-05 at 1 49 50 PM" src="https://github.com/user-attachments/assets/090d4b01-92d6-426a-84ae-1e75c8c4066b" />


8. Specific versions of torch (and other libraries) were installed for Mac with Intel chips: `pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1`
