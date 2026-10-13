<#
.SYNOPSIS
    Menyiapkan artefak sebelum layanan: validasi -> split -> augmentasi ->
    fine-tuning -> seed jawaban -> indeks PDF.

.DESCRIPTION
    Berkas contoh. Jalankan HANYA saat data asli sudah tersedia.
    Gunakan -SkipTrain bila model sudah ada, -SkipIndex bila hanya ingin
    menyegarkan jawaban baku.

.EXAMPLE
    .\deploy\build_artifacts.ps1
    .\deploy\build_artifacts.ps1 -SkipTrain -SkipIndex
#>
[CmdletBinding()]
param(
    [switch]$SkipTrain,
    [switch]$SkipIndex
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Tidak menemukan virtualenv: $python"
}

function Invoke-Py {
    param([string[]]$PyArgs)
    Write-Host "[deploy] python $($PyArgs -join ' ')"
    & $python @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "Perintah gagal: $($PyArgs -join ' ')" }
}

if (-not $SkipTrain) {
    Invoke-Py @("-m", "src.validate_data")
    Invoke-Py @("-m", "src.split")
    Invoke-Py @("-m", "src.augment")
    Invoke-Py @("-m", "src.leakage_check")
    Invoke-Py @("-m", "src.train", "--mode", "single", "--grid", "reduced",
                "--model", "cahya/distilbert-base-indonesian")
}

Invoke-Py @("-m", "scripts.seed_responses")

if (-not $SkipIndex) {
    Invoke-Py @("-m", "rag.index", "--pdf-dir", "data/pdf", "--out", "vectorstore")
}

Invoke-Py @("-m", "scripts.preflight")
Write-Host "[deploy] Artefak selesai disiapkan."
