# EBS volume setup

The S2 cache volume is created once per year, used for 1–2 days, snapshotted, then deleted.
The snapshot persists cheaply (~$2.50/month for 500 GB) and is restored next year — at which
point only new scenes need syncing.

## Why EBS rather than local disk

The pipeline runs on EC2. The instance root volume is typically 50–100 GB — not enough.
A larger root volume could be provisioned, but:

- You pay for it year-round even when idle
- You cannot snapshot just the S2 cache independently of the OS
- You cannot detach it and reattach to a different instance next year

EBS solves all three: ~$40/month while attached, snapshot for pennies/month, delete the
volume when done, restore in minutes next year.

---

## First-time setup

### 1. Find your instance's availability zone

EBS volumes must be in the same AZ as the EC2 instance.

```bash
curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone
# e.g. ap-southeast-2a
```

### 2. Create the volume

```bash
aws ec2 create-volume \
  --availability-zone ap-southeast-2a \
  --size 500 \
  --volume-type gp3 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=s2-cache}]'
```

Note the `VolumeId` in the output (e.g. `vol-0abc123`).

### 3. Attach to your running instance

```bash
aws ec2 attach-volume \
  --volume-id vol-0abc123 \
  --instance-id i-0your-instance \
  --device /dev/sdf
```

### 4. Format (first time only)

```bash
# Check what device name the OS assigned — usually /dev/nvme1n1 on modern instances
lsblk

sudo mkfs -t xfs /dev/nvme1n1
```

### 5. Mount

```bash
sudo mkdir -p /mnt/s2cache
sudo mount /dev/nvme1n1 /mnt/s2cache
sudo chown $(whoami) /mnt/s2cache
```

---

## After the pipeline run — snapshot and delete

```bash
# 1. Snapshot (retained cheaply for next year)
aws ec2 create-snapshot \
  --volume-id vol-0abc123 \
  --description "s2-cache-2025"

# 2. Detach
aws ec2 detach-volume --volume-id vol-0abc123

# 3. Delete
aws ec2 delete-volume --volume-id vol-0abc123
```

---

## Subsequent years — restore from snapshot

```bash
# 1. Create a new volume from last year's snapshot
aws ec2 create-volume \
  --availability-zone ap-southeast-2a \
  --snapshot-id snap-0abc123 \
  --volume-type gp3

# 2. Attach and mount as above (steps 3 and 5 — skip mkfs, data is already there)
```

Only new scenes for the current year need syncing — the snapshot already contains prior
years' data.

---

## Storage sizing

| Band | Resolution | ~Size/file | Files (30 granules × 10 dates) | Total |
|---|---|---|---|---|
| red, nir, green | 10 m | ~300 MB | 900 | ~270 GB |
| rededge1, rededge2 | 20 m | ~80 MB | 600 | ~48 GB |
| scl | 20 m | ~30 MB | 300 | ~9 GB |
| **Total** | | | | **~330 GB** |

A 500 GB gp3 volume provides ~170 GB headroom. If cloud cover is low and scene count
reaches 15–20 dates per granule the total can approach 450–500 GB — size up to 600 GB
if in doubt.
