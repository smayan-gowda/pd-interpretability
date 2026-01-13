# current execution status

## chameleon cloud instance
- **status**: active and running
- **gpu**: quadro rtx 6000 (24gb vram)
- **utilization**: 95% gpu, 3.1gb vram used, 65°c
- **ssh**: `ssh -i ~/.ssh/neuroscope-key cc@192.5.86.214`

## completed phases
### phase 01: setup and data
- **status**: complete and verified
- **duration**: ~3 minutes
- **outputs**:
  - `results/fig_p1_01_dataset_statistics.{pdf,png,svg}`
  - `results/executed_notebooks/01_setup_and_data.ipynb`
- **verification**: dataset loaded (1662 wav files), environment verified, proper latex rendering confirmed (computer modern fonts: CMBX12, CMMI10, CMR10, CMR12, CMR7, CMSY10, CMSY7)

## setup complete
- automatic syncing enabled (30s interval)
- latex rendering verified with proper computer modern fonts
- ready to start phase 03 training

## pending phases
1. **phase 03**: wav2vec2 training (8-12 hours, loso cross-validation with 51 subjects)
2. **phase 04**: activation extraction (1-2 hours)
3. **phase 07**: probing experiments (2-3 hours)
4. **phase 05**: activation patching (3-4 hours)
5. **phase 08**: interpretable prediction (30 min)
6. **phase 06**: cross-dataset generalization (10-15 hours, optional)

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
1. start phase 03 training (awaiting user confirmation)
2. monitor progress with automatic syncing
3. execute remaining phases sequentially after phase 03 completes
4. all results automatically synced to local workspace

## automatic syncing
- script running in background: `scripts/auto_sync_from_chameleon.sh`
- sync interval: 30 seconds
- log file: `/tmp/chameleon_sync.log`
- stop with: `scripts/stop_auto_sync.sh`

## notes
- training is using mixed precision (fp16) for memory efficiency
- checkpoints saved every 10 epochs to `results/checkpoints/wav2vec2_loso_<timestamp>/`
- all 7 figures for phase 03 will be generated at end
- loso cross-validation trains on 51 subjects (leave-one-subject-out)

---
**last updated**: 2026-01-13 07:20 utc
