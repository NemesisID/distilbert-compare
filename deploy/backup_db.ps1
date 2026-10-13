<#
.SYNOPSIS
    Menyalin cadangan database SQLite (bertanggal) ke backups/.

.DESCRIPTION
    Berkas contoh. Untuk database yang sedang aktif dipakai layanan, hentikan
    penulisan terlebih dahulu atau gunakan mekanisme backup SQLite yang aman.

.EXAMPLE
    .\deploy\backup_db.ps1
    .\deploy\backup_db.ps1 -Database data\chatbot.db -OutDir backups
#>
[CmdletBinding()]
param(
    [string]$Database = "data\chatbot.db",
    [string]$OutDir = "backups"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$dbPath = Join-Path $root $Database
if (-not (Test-Path $dbPath)) {
    throw "Database tidak ditemukan: $dbPath"
}

$outPath = Join-Path $root $OutDir
if (-not (Test-Path $outPath)) {
    New-Item -ItemType Directory -Path $outPath | Out-Null
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$name = [System.IO.Path]::GetFileNameWithoutExtension($dbPath)
$dest = Join-Path $outPath "$name`_$stamp.db"

Copy-Item -Path $dbPath -Destination $dest
Write-Host "[deploy] Cadangan dibuat: $dest"
