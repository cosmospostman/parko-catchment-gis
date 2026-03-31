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

### 1. Set variables

```bash
INSTANCE_ID=i-0your-instance
VOLUME_ID=vol-0abc123   # set after step 2
```

### 2. Find your instance's availability zone

EBS volumes must be in the same AZ as the EC2 instance.

```bash
curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone
# e.g. us-west-2a
```

### 3. Create the volume

```bash
aws ec2 create-volume \
  --region us-west-2 \
  --availability-zone us-west-2a \
  --size 500 \
  --volume-type gp3 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=s2-cache}]'
```

Set `VOLUME_ID` to the `VolumeId` in the output, then continue.

### 4. Attach to your running instance

```bash
aws ec2 attach-volume \
  --region us-west-2 \
  --volume-id $VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/sdf
```

### 5. Format (first time only)

```bash
# Check what device name the OS assigned — usually /dev/nvme1n1 on modern instances
lsblk

sudo mkfs -t xfs /dev/nvme1n1
```

### 6. Mount

```bash
sudo mkdir -p /mnt/s2cache
sudo mount /dev/nvme1n1 /mnt/s2cache
sudo chown $(whoami) /mnt/s2cache
```

---

## After the pipeline run — snapshot and delete

```bash
VOLUME_ID=vol-0abc123   # set to your volume

# 1. Snapshot (retained cheaply for next year)
aws ec2 create-snapshot \
  --region us-west-2 \
  --volume-id $VOLUME_ID \
  --description "s2-cache-2026"

# 2. Detach
aws ec2 detach-volume --region us-west-2 --volume-id $VOLUME_ID

# 3. Delete
aws ec2 delete-volume --region us-west-2 --volume-id $VOLUME_ID
```

---

## Subsequent years — restore from snapshot

```bash
INSTANCE_ID=i-0your-instance

# 1. Create a new volume from last year's snapshot
aws ec2 create-volume \
  --region us-west-2 \
  --availability-zone us-west-2a \
  --snapshot-id snap-0abc123 \
  --volume-type gp3

VOLUME_ID=vol-0new-volume   # set from output above

# 2. Attach and mount as above (steps 4 and 6 — skip mkfs, data is already there)
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
