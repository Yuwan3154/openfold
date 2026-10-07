#!/bin/bash
#SBATCH -J esm2_envprobe
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/home/gridsan/cou/logs/esm2_envprobe-%j.out
source /etc/profile
module load conda/Python-ML-2025b-pytorch
for e in esm cathfold proteina pottsmpnn protebm protpardelle; do
  source activate $e 2>/dev/null
  echo "== $e: $(python -c '
import importlib
out=[]
for m in ("esm","transformers","flash_attn","torch"):
    try:
        x=importlib.import_module(m); out.append(m+"="+getattr(x,"__version__","?"))
    except Exception as ex: out.append(m+"=MISSING")
try:
    from esm.models.esmfold2 import ESMFold2InputBuilder; out.append("builder=ok")
except Exception as ex: out.append("builder=NO")
try:
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model; out.append("model=ok")
except Exception as ex: out.append("model=NO")
print(" ".join(out))' 2>&1 | tail -1)"
done
ls ~/.cache/huggingface/hub 2>/dev/null | head
