param(
    [Parameter(Mandatory = $true)]
    [int]$ScpProcessId,
    [Parameter(Mandatory = $true)]
    [string]$CheckpointDir,
    [Parameter(Mandatory = $true)]
    [string]$PathInRepo
)

$ErrorActionPreference = 'Stop'
$logPath = Join-Path (Split-Path -Parent $CheckpointDir) 'hf_upload.log'

Wait-Process -Id $ScpProcessId

$modelPath = Join-Path $CheckpointDir 'model\model.safetensors'
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "scp finished without a model file: $modelPath"
}

Set-Location (Join-Path $PSScriptRoot '..')
uv run --with huggingface_hub python tools/upload_hf_checkpoint.py $CheckpointDir `
    --repo-id PhangHongHao/UFO-Z1 --path-in-repo $PathInRepo *>&1 | Tee-Object -FilePath $logPath
