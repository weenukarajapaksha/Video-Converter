# Sets up the isolated Wav2Lip environment used by the optional `lipsync` stage.
#   powershell -ExecutionPolicy Bypass -File scripts\setup_lipsync.ps1
# Creates: third_party\Wav2Lip (pinned commit + checkpoints) and .venv-lipsync (Python 3.11).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Repo = Join-Path $Root "third_party\Wav2Lip"
$Venv = Join-Path $Root ".venv-lipsync"
$Commit = "bac9a81e63ecc153202353372e5724b83d9e6322"
$TorchIndex = if ($env:LIPSYNC_TORCH_INDEX) { $env:LIPSYNC_TORCH_INDEX } else { "https://download.pytorch.org/whl/cu126" }

if (-not (Test-Path (Join-Path $Repo ".git"))) {
    git clone https://github.com/Rudrabha/Wav2Lip $Repo
}
git -C $Repo fetch --depth 1 origin $Commit 2>$null
git -C $Repo checkout -q $Commit

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    py -3.11 -m venv $Venv
}
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install -q --upgrade pip
& $Py -m pip install -q torch==2.7.1 --index-url $TorchIndex
& $Py -m pip install -q "numpy<2.3" opencv-python-headless==4.10.0.84 librosa==0.10.2.post1 scipy

$Downloads = @{
    "checkpoints\wav2lip_gan.pth" = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
    "face_detection\detection\sfd\s3fd.pth" = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
}
foreach ($rel in $Downloads.Keys) {
    $dst = Join-Path $Repo $rel
    if (-not (Test-Path $dst)) {
        New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
        Write-Host "downloading $rel"
        curl.exe -L --fail -o $dst $Downloads[$rel]
    }
}

& $Py -c "import torch, cv2, librosa; print('lipsync env ok: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
