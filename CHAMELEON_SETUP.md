# chameleon cloud setup complete

## instance details
- **ip**: 192.5.86.214
- **gpu**: quadro rtx 6000 (24gb vram)
- **cuda**: 12.6
- **pytorch**: 2.9.1+cu126
- **storage**: 66gb available

## environment setup
- python 3.12 with venv at `~/projects/pd-interpretability/venv`
- all dependencies installed (pytorch, transformers, librosa, etc.)
- ffmpeg and audio libraries installed
- jupyter server configured
- all notebooks converted from colab to chameleon format

## directory structure
```
/home/cc/projects/pd-interpretability/
├── data/
│   └── raw/
│       └── italian_pvs/       # 817mb, 1662 wav files
├── notebooks/
│   └── gpu/                   # all converted notebooks
├── results/
│   ├── executed_notebooks/    # output from nbconvert
│   ├── checkpoints/           # model checkpoints
│   └── figures/               # generated visualizations
├── venv/                      # python virtual environment
└── scripts/
    └── convert_to_chameleon.py
```

## accessing the instance

### ssh access
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214
```

### jupyter access
jupyter is running in tmux session "jupyter"

**via ssh tunnel** (recommended):
```bash
# on local machine
ssh -i ~/.ssh/neuroscope-key -L 8888:localhost:8888 cc@192.5.86.214 -N
# then open: http://localhost:8888
# token: 73c2b875cdbe6a6167ed011fcd37344be49be43dffe88163
```

**direct access** (if firewall allows):
```
http://192.5.86.214:8888/?token=73c2b875cdbe6a6167ed011fcd37344be49be43dffe88163
```

### tmux sessions
- `jupyter` - jupyter notebook server
- `phase01` - currently running phase 01 notebook

attach to session:
```bash
tmux attach -t <session_name>
```

detach from session: `ctrl+b` then `d`

## utility scripts

### start jupyter
```bash
~/.local/bin/start-jupyter.sh
```

### run notebook with monitoring
```bash
~/.local/bin/run-notebook.sh <notebook_name> [session_name]
# example:
~/.local/bin/run-notebook.sh 03_train_wav2vec2.ipynb training
```

creates tmux session with:
- left pane: notebook execution
- right pane: gpu monitoring (nvidia-smi)

## execution order

### current status
- **phase 01**: running now (setup and data verification)

### next steps
1. **phase 03**: wav2vec2 training (8-12 hours, critical)
2. **phase 04**: activation extraction (1-2 hours)
3. **phase 07**: probing experiments (2-3 hours)
4. **phase 05**: activation patching (3-4 hours)
5. **phase 08**: interpretable prediction (30 min)
6. **phase 06**: cross-dataset generalization (10-15 hours, optional)

### manual execution
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214
cd ~/projects/pd-interpretability
source venv/bin/activate

# run notebook
jupyter nbconvert --to notebook --execute \
  notebooks/gpu/<notebook>.ipynb \
  --output-dir results/executed_notebooks \
  --ExecutePreprocessor.timeout=86400
```

## monitoring

### check running sessions
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214 "tmux ls"
```

### view notebook progress
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214 "tmux capture-pane -t phase01 -p | tail -50"
```

### gpu utilization
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214 "nvidia-smi"
```

### check disk space
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214 "df -h /"
```

## results transfer

### download results
```bash
# sync all results
rsync -avz -e "ssh -i ~/.ssh/neuroscope-key" \
  cc@192.5.86.214:~/projects/pd-interpretability/results/ \
  "/Volumes/usb drive/pd-interpretability/results/"

# sync specific figures
rsync -avz -e "ssh -i ~/.ssh/neuroscope-key" \
  cc@192.5.86.214:~/projects/pd-interpretability/results/fig_*.{pdf,png,svg} \
  "/Volumes/usb drive/pd-interpretability/results/"
```

### upload to repo
```bash
cd "/Volumes/usb drive/pd-interpretability"
git add results/
git commit -m "add results from chameleon cloud execution"
git push origin main
```

## troubleshooting

### restart jupyter
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214
tmux kill-session -t jupyter
~/.local/bin/start-jupyter.sh
```

### check notebook errors
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214
cd ~/projects/pd-interpretability
cat results/executed_notebooks/<notebook_name>_executed.ipynb | grep -A5 "error"
```

### free up disk space
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214
# clean old checkpoints
rm -rf ~/projects/pd-interpretability/results/checkpoints/old_*
# clean cache
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface
```

### kill stuck process
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214
tmux kill-session -t <session_name>
```

## notes
- all notebooks have been converted to use `/home/cc/projects/pd-interpretability` paths
- google colab imports and drive mounting removed
- gpu has 24gb vram (note: notebooks expect 48gb for rtx 6000, may need batch size adjustments)
- training uses mixed precision (fp16) and gradient checkpointing to reduce memory
- checkpoints saved every 10 epochs
- all visualizations save to pdf/png/svg formats

## estimated execution times
- phase 01: 5 min
- phase 03: 8-12 hours (loso cross-validation, 51 subjects)
- phase 04: 1-2 hours
- phase 05: 3-4 hours
- phase 06: 10-15 hours (lodo cross-dataset)
- phase 07: 2-3 hours
- phase 08: 30 min

**total**: ~25-35 hours for full pipeline
