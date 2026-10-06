#!/bin/bash
# A6000 side of the Engaging unique-data move. Pulls the 3 tars from the SuperCloud hub, verifies, extracts, re-verifies.
# Needs ~/e_clean/expected_sha.txt ("<sha256>  <name>.tar", from Engaging's stream sha) and ~/e_clean/ckpt_sha256.txt
# (per-ckpt sha256 with of_run-relative paths). EXPECT_COUNTS="runD_logs=58 runD_t4pool=751494 misc=<n>" as env.
# NOTHING on Engaging is touched; the A6000 copy must verify before any deletion is even considered.
set -u
D=$HOME/e_clean/engaging_unique
mkdir -p $D/tars $D/of_run_unique
nice -n 10 ionice -c3 rsync -a --partial --info=stats1 SuperCloud:e_clean/xfer/ $D/tars/ || { echo "rsync failed"; echo rc_final=1; exit 1; }
rc=0
cd $D/tars
echo "== sha256 of the received tars vs the stream sha computed on Engaging"
nice -n 10 ionice -c3 sha256sum *.tar > $D/received_sha.txt
while read -r sha name; do
  name=${name#\*}
  got=$(grep -F " $name" $D/received_sha.txt | awk '{print $1}' | head -1)
  [ "$got" = "$sha" ] && echo "OK    $name" || { echo "DIFF  $name expected=$sha got=$got"; rc=1; }
done < $HOME/e_clean/expected_sha.txt
echo "== member counts (files only) vs the manifests"
for kv in $EXPECT_COUNTS; do
  n=${kv%%=*}; want=${kv##*=}
  got=$(tar tf $n.tar | grep -vc '/$')
  [ "$got" = "$want" ] && echo "OK    $n members=$got" || { echo "DIFF  $n members=$got want=$want"; rc=1; }
done
echo "== extract runD_logs and misc, verify every ckpt sha256 against Engaging's per-file list"
for t in runD_logs misc; do nice -n 10 ionice -c3 tar xf $t.tar -C $D/of_run_unique || rc=1; done
cd $D/of_run_unique
bad=$(nice -n 10 ionice -c3 sha256sum -c $HOME/e_clean/ckpt_sha256.txt 2>&1 | grep -vc ': OK$')
echo "ckpt sha256 -c: files=$(wc -l < $HOME/e_clean/ckpt_sha256.txt) not-OK=$bad"
[ "$bad" = 0 ] || rc=1
df -h $HOME | tail -1
echo "rc_final=$rc"
exit $rc
