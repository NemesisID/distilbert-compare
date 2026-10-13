<#
.SYNOPSIS
    Menjalankan layanan API chatbot di latar depan (BUKAN layanan permanen).

.DESCRIPTION
    Berkas contoh. Skrip ini hanya memanggil uvicorn; hentikan dengan Ctrl+C.
    Tidak memasang layanan Windows, tidak mengubah firewall/registry/IIS.

.EXAMPLE
    .\deploy\run_api.ps1
    .\deploy\run_api.ps1 -Port 8080 -Reload
#>
[CmdletBinding()]
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONUTF8 = "1"

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Tidak menemukan virtualenv: $python. Buat dulu lalu pasang dependensi."
}

$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) {
    throw ".env belum ada. Salin .env.example -> .env lalu isi CHATBOT_API_KEY."
}

Write-Host "[deploy] Pra-terbang (preflight) ..."
& $python -m scripts.preflight
if ($LASTEXITCODE -ne 0) {
    throw "Preflight GAGAL. Lengkapi artefak sebelum menjalankan layanan."
}

$uvArgs = @("-m", "uvicorn", "api.main:app", "--host", $BindHost, "--port", "$Port")
if ($Reload) { $uvArgs += "--reload" }

Write-Host "[deploy] uvicorn -> http://${BindHost}:$Port  (Ctrl+C untuk berhenti)"
& $python @uvArgs
exit $LASTEXITCODE
