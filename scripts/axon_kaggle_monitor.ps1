param(
    [Parameter(Mandatory = $true)]
    [string]$JobId
)

$ErrorActionPreference = "Stop"
$AxonRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $AxonRoot

Write-Host "Axon Kaggle Live Monitor" -ForegroundColor Cyan
Write-Host "Job: $JobId"
Write-Host "Closing this terminal never stops Kaggle training." -ForegroundColor Yellow
Write-Host ""
& python "$AxonRoot\scripts\axon_kaggle.py" monitor $JobId --follow
$ExitCode = $LASTEXITCODE
Write-Host ""
if ($ExitCode -eq 0) {
    Write-Host "The Kaggle log stream ended cleanly." -ForegroundColor Green
} else {
    Write-Host "The monitor ended with code $ExitCode. The cloud job may still be running." -ForegroundColor Yellow
    & python "$AxonRoot\scripts\axon_kaggle.py" status $JobId
}
