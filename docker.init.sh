#!/bin/bash
set -eo pipefail

# PySide6's Qt xcb platform plugin depends on the native X11/XCB runtime stack.
# Minimal Vast/Runpod images frequently omit part of that stack, causing Qt to
# find libqxcb.so successfully but abort while loading one of its shared-library
# dependencies.
if command -v apt-get >/dev/null 2>&1; then
  qt_xcb_packages=(
    libx11-6
    libx11-xcb1
    libxcb1
    libxcb-cursor0
    libxcb-icccm4
    libxcb-image0
    libxcb-keysyms1
    libxcb-randr0
    libxcb-render0
    libxcb-render-util0
    libxcb-shape0
    libxcb-shm0
    libxcb-sync1
    libxcb-util1
    libxcb-xfixes0
    libxcb-xkb1
    libxkbcommon0
    libxkbcommon-x11-0
    libxrender1
    libxext6
    libxi6
    libsm6
    libice6
    libfontconfig1
    libfreetype6
    libgl1
  )

  missing_qt_xcb_packages=()
  for package in "${qt_xcb_packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
      missing_qt_xcb_packages+=("$package")
    fi
  done

  if ((${#missing_qt_xcb_packages[@]})); then
    echo "Installing Qt X11/xcb runtime dependencies..."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      "${missing_qt_xcb_packages[@]}"
    rm -rf /var/lib/apt/lists/*
  fi
fi

# Export useful ENV variables, including all Runpod specific vars, to /etc/rp_environment
echo "Exporting environment variables..."
printenv |
  grep -E '^RUNPOD_|^PATH=|^HF_HOME=|^HF_TOKEN=|^HUGGING_FACE_HUB_TOKEN=|^WANDB_API_KEY=|^WANDB_TOKEN=' |
  sed 's/^\(.*\)=\(.*\)$/export \1="\2"/' >> /etc/rp_environment

# Add it to Bash login script only if it doesn't already exist
grep -qxF 'source /etc/rp_environment' ~/.bashrc || echo 'source /etc/rp_environment' >> ~/.bashrc
echo "cd /workspace/OneTrainer" >> ~/.bashrc

source /etc/rp_environment

# Vast.ai uses $SSH_PUBLIC_KEY
if [[ $SSH_PUBLIC_KEY ]]; then
  echo "INFO: Found SSH_PUBLIC_KEY, using it as PUBLIC_KEY"
  PUBLIC_KEY="${SSH_PUBLIC_KEY}"
fi

# Runpod uses $PUBLIC_KEY
if [[ $PUBLIC_KEY ]]; then
  echo "INFO: Setting up SSH, adding PUBLIC_KEY to authorized_keys"
  mkdir -p ~/.ssh
  echo "${PUBLIC_KEY}" >> ~/.ssh/authorized_keys
  chmod 600 ~/.ssh/authorized_keys
  chmod 700 ~/.ssh
fi

# disable SSH password login - use key instead!
sed -i -E 's/#?PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Start SSH server
service ssh start 2>&1

# Login to HF
if [[ -n "${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" ]]; then
  pixi run --locked -e ${OT_PLATFORM} hf auth login --token "${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential 2>&1
else
  echo "HF_TOKEN or HUGGING_FACE_HUB_TOKEN not set; skipping login"
fi

# Login to WanDB
if [[ -n "${WANDB_API_KEY:-$WANDB_TOKEN}" ]]; then
  pixi run --locked -e ${OT_PLATFORM} wandb login "${WANDB_API_KEY:-$WANDB_TOKEN}" 2>&1
else
  echo "WANDB_API_KEY or WANDB_TOKEN not set; skipping login"
fi

mkdir -p /workspace
ln -s /OneTrainer /workspace/OneTrainer

# Keep the container running
sleep infinity
