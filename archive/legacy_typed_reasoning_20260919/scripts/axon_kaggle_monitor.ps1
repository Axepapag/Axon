param(
    [string]$JobId = ""
)

$ErrorActionPreference = "Stop"
$AxonRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $AxonRoot

Write-Host "Axon Kaggle Live Monitor" -ForegroundColor Cyan
Write-Host "Closing this terminal never stops Kaggle training." -ForegroundColor Yellow
Write-Host ""

$MonitorArgs = @("$AxonRoot\scripts\axon_kaggle.py", "monitor")
if ($JobId) {
    $MonitorArgs += $JobId
    Write-Host "Job: $JobId"
} else {
    Write-Host "Listing jobs on this machine. Choose a number; you do not need to paste an ID."
}
$MonitorArgs += "--follow"
& python @MonitorArgs
$ExitCode = $LASTEXITCODE
Write-Host ""
if ($ExitCode -eq 0) {
    Write-Host "The Kaggle log stream ended cleanly." -ForegroundColor Green
} else {
    Write-Host "The monitor ended with code $ExitCode. The cloud job may still be running." -ForegroundColor Yellow
    if ($JobId) {
        & python "$AxonRoot\scripts\axon_kaggle.py" status $JobId
    }
}
