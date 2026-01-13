# current execution status

## chameleon cloud instance
- **status**: active and running
- **gpu**: quadro rtx 6000 (24gb vram)
- **utilization**: 95% gpu, 3.1gb vram used, 65°c
- **ssh**: `ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214`

## completed phases
### phase 01: setup and data
- **status**: complete
- **duration**: ~3 minutes
- **outputs**: 
  - `results/fig_p1_01_dataset_statistics.{pdf,png,svg}`
  - `results/executed_notebooks/01_setup_and_data.ipynb`
- **verification**: dataset loaded (1662 wav files), environment verified, figure generated

## currently running
### phase 03: wav2vec2 training (loso cross-validation)
- **status**: in progress
- **started**: 2026-01-13 07:08 utc
- **estimated duration**: 8-12 hours
- **tmux session**: `training`
- **monitor**: 
  - left pane: notebook execution
  - right pane: gpu monitoring
- **attach**: `ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214 -t "tmux attach -t training"`

## pending phases
1. **phase 04**: activation extraction (1-2 hours)
2. **phase 07**: probing experiments (2-3 hours)
3. **phase 05**: activation patching (3-4 hours)
4. **phase 08**: interpretable prediction (30 min)
5. **phase 06**: cross-dataset generalization (10-15 hours, optional)

## monitoring commands

### check training progress
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214 "tmux capture-pane -t training.left -p | tail -50"
```

### check gpu status
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214 "nvidia-smi"
```

### list tmux sessions
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214 "tmux ls"
```

### check disk space
```bash
ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214 "df -h /"
```

## jupyter access
- **url**: http://localhost:8888 (via tunnel)
- **token**: 73c2b875cdbe6a6167ed011fcd37344be49be43dffe88163
- **tunnel**: `ssh -i ~/.ssh/neuroscope-key -L 8888:localhost:8888 cc@192.5.86.214 -N`

## next steps
1. wait for phase 03 to complete (check every few hours)
2. download phase 03 results and figures
3. execute remaining phases sequentially
4. sync all results back to local machine
5. commit and push to repository

## estimated completion time
- phase 03 should complete by: ~2026-01-13 15:00-19:00 utc (8-12 hours from start)
- full pipeline: ~25-35 hours total

## notes
- training is using mixed precision (fp16) for memory efficiency
- checkpoints saved every 10 epochs to `results/checkpoints/wav2vec2_loso_<timestamp>/`
- all 7 figures for phase 03 will be generated at end
- loso cross-validation trains on 51 subjects (leave-one-subject-out)

---
**last updated**: 2026-01-13 07:11 utc
