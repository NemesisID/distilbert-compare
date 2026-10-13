<#
.SYNOPSIS
    Memeriksa endpoint /health layanan chatbot.

.EXAMPLE
    .\deploy\healthcheck.ps1
    .\deploy\healthcheck.ps1 -BaseUrl http://127.0.0.1:8080
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"

try {
    $resp = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -TimeoutSec 10
    Write-Host "[deploy] /health OK: $($resp | ConvertTo-Json -Compress)"
    exit 0
}
catch {
    Write-Error "[deploy] /health GAGAL: $($_.Exception.Message)"
    exit 1
}
