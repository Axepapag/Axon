$ErrorActionPreference = "Continue"
$AxonRoot = Split-Path -Parent $PSScriptRoot
$KaggleCli = "$AxonRoot\scripts\axon_kaggle.py"
$DefaultRecipe = "$AxonRoot\configs\kaggle\d64_first_cloud_tranche.json"
Set-Location -LiteralPath $AxonRoot

function Invoke-AxonKaggle {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & python $KaggleCli @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Stopped safely. Nothing later in this action was attempted." -ForegroundColor Red
    }
}

while ($true) {
    Clear-Host
    Write-Host "============================================================" -ForegroundColor DarkCyan
    Write-Host "              AXON KAGGLE CONTROL CENTER" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor DarkCyan
    Write-Host "Training runs on Kaggle, not inside Codex or this window."
    Write-Host "Closing this window does not stop a submitted cloud job." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1  Check login and remaining GPU/TPU time"
    Write-Host "2  List Axon cloud jobs known on this machine"
    Write-Host "3  Prepare the first D64 Kaggle packet (NO upload)"
    Write-Host "4  Launch or retry a packet privately on Kaggle"
    Write-Host "5  Open a separate live monitor window"
    Write-Host "6  Check one job now"
    Write-Host "7  Download completed outputs"
    Write-Host "8  Open Axon's private Kaggle kernels page"
    Write-Host "Q  Quit"
    Write-Host ""
    $Choice = (Read-Host "Choose").Trim().ToUpperInvariant()
    switch ($Choice) {
        "1" {
            Invoke-AxonKaggle doctor
        }
        "2" {
            Invoke-AxonKaggle jobs
        }
        "3" {
            Write-Host "Preparing from:" $DefaultRecipe -ForegroundColor Cyan
            Write-Host "This creates a local, hashed packet and uploads nothing." -ForegroundColor Green
            Invoke-AxonKaggle prepare $DefaultRecipe
        }
        "4" {
            $JobId = (Read-Host "Paste the full prepared job ID").Trim()
            if ($JobId) {
                Write-Host "This uploads a PRIVATE dataset if needed and submits a PRIVATE Kaggle job version." -ForegroundColor Yellow
                Write-Host "An active running/queued job will be detected and will not be duplicated." -ForegroundColor Yellow
                Write-Host "It may consume Kaggle accelerator quota." -ForegroundColor Yellow
                $Confirm = (Read-Host "Type LAUNCH to authorize this exact job").Trim()
                if ($Confirm -ceq "LAUNCH") {
                    Invoke-AxonKaggle launch $JobId --yes
                    if ($LASTEXITCODE -eq 0) {
                        Start-Process powershell.exe -ArgumentList @(
                            "-NoExit",
                            "-ExecutionPolicy", "Bypass",
                            "-File", "$AxonRoot\scripts\axon_kaggle_monitor.ps1",
                            "-JobId", $JobId
                        )
                    }
                } else {
                    Write-Host "Launch cancelled. No upload or quota spend occurred." -ForegroundColor Green
                }
            }
        }
        "5" {
            $JobId = (Read-Host "Paste the full job ID").Trim()
            if ($JobId) {
                Start-Process powershell.exe -ArgumentList @(
                    "-NoExit",
                    "-ExecutionPolicy", "Bypass",
                    "-File", "$AxonRoot\scripts\axon_kaggle_monitor.ps1",
                    "-JobId", $JobId
                )
            }
        }
        "6" {
            $JobId = (Read-Host "Paste the full job ID").Trim()
            if ($JobId) { Invoke-AxonKaggle status $JobId }
        }
        "7" {
            $JobId = (Read-Host "Paste the full completed job ID").Trim()
            if ($JobId) { Invoke-AxonKaggle fetch $JobId }
        }
        "8" {
            Start-Process "https://www.kaggle.com/code/axongliksbot"
        }
        "Q" { break }
        default { Write-Host "Unknown choice." -ForegroundColor Yellow }
    }
    if ($Choice -eq "Q") { break }
    Write-Host ""
    Read-Host "Press Enter to return to the menu"
}
