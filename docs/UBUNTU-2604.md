# Fresh install setup — Ubuntu 26.04

This machine is an HP Z640 with an RTX 5060 Ti (Blackwell/sm_120, 16GB VRAM), 62GB RAM,
and two LVM volumes: 98GB `/` and 369GB `/data`.

---

## 1. Disk layout

During the Ubuntu installer, recreate the LVM layout:

- `vg0/lv-0` → 98GB → `/`
- `vg0/lv-1` → 369GB → `/data`
- EFI partition on nvme (`/dev/nvme0n1p1`) → `/boot/efi`

**Do not format `/data`** if it already contains the data backup — just mount it.

---

## 2. First boot

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install git curl vim rsync python3.14-venv nload avahi
```

Set hostname:

```bash
sudo hostnamectl set-hostname z640
```

---

## 3. Restore home directory

```bash
sudo mkdir -p /home/mlj
sudo chown mlj:mlj /home/mlj
rsync -av /data/home-backup/ /home/mlj/
```

This restores `.ssh`, `.gitconfig`, `.bashrc`, `.claude`, `parko-catchment-gis/`, `weeds/`, etc.

---

## 4. NVIDIA driver

The RTX 5060 Ti (Blackwell, sm_120) requires the open kernel module driver:

```bash
sudo apt install nvidia-driver-595-open
sudo reboot
```

Verify after reboot:

```bash
nvidia-smi
```

---

## 5. GDAL / PROJ system packages

```bash
sudo apt install gdal-bin libgdal-dev libproj-dev python3-gdal
```

---

## 6. Python environment

Ubuntu 26.04 ships Python 3.12. The project uses a venv at `~/parko-catchment-gis/.venv`
which will be restored from the home backup — but the venv may need rebuilding if paths changed.

```bash
cd ~/parko-catchment-gis
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch — Blackwell GPU requires cu130 + PyTorch 2.7+

The RTX 5060 Ti (sm_120) requires CUDA 13.x. Use the cu130 index (latest available):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Verify:

```bash
python3 -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
# expect: 12.8 / True
```

---

## 7. SSH key

The `.ssh` directory is restored from the home backup. Fix permissions:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_* ~/.ssh/authorized_keys 2>/dev/null
chmod 644 ~/.ssh/*.pub ~/.ssh/known_hosts 2>/dev/null
```

Test GitHub access:

```bash
ssh -T git@github.com
```

---

## 8. /data volume contents

The `/data` volume (369GB, ~186GB used at backup time) contains:

| Path | Size | Notes |
|---|---|---|
| `/data/pixels` | 79GB | Primary raster data |
| `/data/chips` | 44GB | Training chips |
| `/data/training` | 24GB | Model training runs |
| `/data/mrc-parko` | 8.2GB | MRC Parkinsonia data |
| `/data/cache` | 4.2GB | Regenerable |
| `/data/tmpfn_09qr__pixel_df.ipc` | 12GB | Delete — stale temp file |
| `/data/tmpjh6okamd_caller_pixel_df.ipc` | 19GB | Delete — stale temp file |

Clean up temp files after confirming data is intact:

```bash
rm /data/tmp*.ipc
```

---

## 9. Verify project

```bash
cd ~/parko-catchment-gis
source .venv/bin/activate
pytest tests/
```
