# EBS volume setup

A single EBS volume mounted at `/mnt/ebs` holds both the Sentinel-2 COG cache (`s2cache/`)
and the Sentinel-1 GRD cache (`s1cache/`). It is created once per year, used for 1–2 days,
snapshotted, then deleted. The snapshot persists cheaply and is restored next year — at which
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
INSTANCE_ID=i-068fc9ec12b1c044e
AZ=us-west-2b
VOLUME_ID=vol-0bbe5d41b37159d7e   # set after step 3
```

### 2. Find your instance's availability zone

EBS volumes must be in the same AZ as the EC2 instance.

```bash
curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone
# e.g. us-west-2b
```

### 3. Create the volume

```bash
aws ec2 create-volume \
  --region us-west-2 \
  --availability-zone $AZ \
  --size 2000 \
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
sudo mkdir -p /mnt/ebs
sudo mount /dev/nvme1n1 /mnt/ebs
sudo chown $(whoami) /mnt/ebs
mkdir -p /mnt/ebs/s2cache /mnt/ebs/s1cache
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
  --availability-zone $AZ \
  --snapshot-id snap-0abc123 \
  --volume-type gp3

VOLUME_ID=vol-0new-volume   # set from output above

# 2. Attach and mount as above (steps 4 and 6 — skip mkfs, data is already there)
```

Only new scenes for the current year need syncing — the snapshot already contains prior
years' data.

---

## EBS throughput

The default gp3 throughput is 125 MB/s, which will be the bottleneck — not the network.
Before starting the sync, increase throughput to the gp3 maximum. The constraint is
`throughput / IOPS ≤ 0.25 MiBps per IOPS`, so 1000 MB/s requires at least 4000 IOPS:

```bash
aws ec2 modify-volume --volume-id $VOLUME_ID --throughput 1000 --iops 4000
```

Takes effect within ~1 minute without unmounting. At 1000 MB/s the sync runs in ~5–10 min
instead of ~40 min. There is no extra cost for gp3 throughput up to 1000 MB/s or IOPS up
to 16000.

You can confirm the volume is no longer the bottleneck with:

```bash
iostat -xm 2 5 /dev/nvme1n1   # wMB/s should be well below %util 100
sar -n DEV 2 5                 # rxkB/s shows actual network throughput
```

---

## Storage sizing

The actual 2025 sync produced 6774 files. The original estimate of ~330 GB (30 granules ×
10 dates) significantly underestimated scene count — actual usage exceeded 500 GB.

Use a **2 TB gp3 volume**. It provides ample headroom for multiple years of data as the
snapshot accumulates scenes year over year.

To resize an existing volume (e.g. if you started with 500 GB):

```bash
# Wait for any in-progress modification to complete first
aws ec2 describe-volumes-modifications --volume-id $VOLUME_ID \
  --query 'VolumesModifications[0].{State:ModificationState,Progress:Progress}'

# Resize (only works when ModificationState is "completed")
aws ec2 modify-volume --volume-id $VOLUME_ID --size 2000

# Expand the filesystem without unmounting
sudo growpart /dev/nvme1n1 1
sudo xfs_growfs /mnt/ebs
```
