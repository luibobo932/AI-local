param(
    [switch]$SkipTrainingPackages
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PIP_PROGRESS_BAR = "off"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $repo ".venv-cuda"
$python = Join-Path $venv "Scripts\python.exe"

try {
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host "Đang tạo môi trường GPU riêng tại $venv"
        python -m venv $venv
    }

    & $python -m pip install --upgrade pip

    # Ghim rõ wheel CUDA 13.0; PyPI mặc định trên máy này trả về bản +cpu.
    & $python -m pip install --upgrade "torch==2.12.1+cu130" `
        --index-url "https://download.pytorch.org/whl/cu130"
    & $python -m pip install -r (Join-Path $repo "requirements.txt")

    if (-not $SkipTrainingPackages) {
        & $python -m pip install -r (Join-Path $repo "requirements-training.txt")
    }

    & $python (Join-Path $repo "verify_minion_gpu.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Môi trường đã tạo nhưng PyTorch chưa nhận GPU."
    }

    Write-Host "Minion đã có môi trường GPU hoạt động: $python"
} catch {
    Write-Error "Không thiết lập được môi trường GPU: $($_.Exception.Message)"
    exit 1
}
