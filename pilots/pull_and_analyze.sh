#!/bin/bash
# Pull sweep results from server and generate summary
SSH_HOST="connect.westc.seetacloud.com"
SSH_PORT="46503"
SSH_PASS="J1AeQodNS/uj"
LOCAL_DIR="/Users/arthuryang/Desktop/research/HSI/experiment_data/sweep"

echo "Pulling results from server..."
mkdir -p "$LOCAL_DIR"

# Find latest sweep directory
LATEST=$(sshpass -p "$SSH_PASS" ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no root@"$SSH_HOST" \
    "ls -dt /root/autodl-tmp/results/sweep_* 2>/dev/null | head -1" 2>/dev/null)

if [ -z "$LATEST" ]; then
    echo "No sweep results found on server"
    exit 1
fi

echo "Latest sweep: $LATEST"

# Pull all result JSONs, logs, configs (skip checkpoints and features)
sshpass -p "$SSH_PASS" ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no root@"$SSH_HOST" \
    "cd $LATEST && find . -name '*.json' -o -name '*.csv' -o -name '*.log' -o -name '*.yaml' | tar czf /tmp/sweep_results.tar.gz -T -" 2>/dev/null

sshpass -p "$SSH_PASS" scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
    root@"$SSH_HOST":/tmp/sweep_results.tar.gz "$LOCAL_DIR/sweep_results.tar.gz" 2>/dev/null

cd "$LOCAL_DIR"
tar xzf sweep_results.tar.gz
rm sweep_results.tar.gz

echo ""
echo "═══ Sweep Results Summary ═══"
for dir in */; do
    name=$(basename "$dir")
    result=$(grep 'CMCD-LoRA+SHINE' "$dir/run.log" 2>/dev/null | tail -1)
    if [ -n "$result" ]; then
        echo "  $name: $result"
    fi
done

echo ""
echo "Results saved to: $LOCAL_DIR"
