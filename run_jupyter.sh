#!/bin/bash
#SBATCH --gres=gpu:1 
#SBATCH --nodes 1
#SBATCH --time 4:00:00
#SBATCH --mem-per-cpu 10G
#SBATCH --job-name tunnel
#SBATCH --output jupyter-log-%J.txt

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -i ~/.ssh/id_tiger -N -L $ipnport:$ipnip:$ipnport $USER@tiger.princeton.edu
    -----------------------------------------------------------------

    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "

jupyter notebook --no-browser --port=$ipnport --ip=$ipnip
