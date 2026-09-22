param(
    [string]$RepoUrl = "https://github.com/Axepapag/Axon.git",
    [string]$Branch = "cloud-vm-control-20260921",
    [string]$AxonRoot = "C:\Axon",
    [string]$StateRoot = "C:\Axon\State",
    [string]$PublicHost = "",
    [int]$ControlPort = 8765
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Require-Command([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Required command '$Name' is not installed or not on PATH." }
    return $cmd.Source
}

$git = Require-Command "git"
$python = Require-Command "python"

if (-not (Test-Path $AxonRoot)) {
    & $git clone --branch $Branch --single-branch $RepoUrl $AxonRoot
} else {
    Push-Location $AxonRoot
    try {
        & $git fetch origin $Branch
        & $git checkout $Branch
        & $git pull --ff-only origin $Branch
    } finally { Pop-Location }
}

$venv = Join-Path $AxonRoot ".venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $python -m venv $venv
}
$venvPython = Join-Path $venv "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e $AxonRoot

New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
$opsRoot = Join-Path $env:ProgramData "Axon"
New-Item -ItemType Directory -Force -Path $opsRoot | Out-Null

$tokenPath = Join-Path $opsRoot "control-token.txt"
if (-not (Test-Path $tokenPath)) {
    $token = (& $venvPython -c "import secrets; print(secrets.token_urlsafe(48))").Trim()
    Set-Content -Path $tokenPath -Value $token -NoNewline -Encoding ascii
    & icacls $tokenPath /inheritance:r /grant:r "SYSTEM:F" "Administrators:F" | Out-Null
}
$token = (Get-Content $tokenPath -Raw).Trim()

$heartCmd = Join-Path $opsRoot "run-heart.cmd"
$controlCmd = Join-Path $opsRoot "run-control.cmd"
@"
@echo off
cd /d "$AxonRoot"
"$venvPython" scripts\run_axon_heart.py --state-root "$StateRoot"
"@ | Set-Content -Path $heartCmd -Encoding ascii

@"
@echo off
set AXON_STATE_ROOT=$StateRoot
set AXON_CONTROL_TOKEN=$token
cd /d "$AxonRoot"
"$venvPython" -m uvicorn scripts.axon_control_server:app --host 127.0.0.1 --port $ControlPort
"@ | Set-Content -Path $controlCmd -Encoding ascii

$taskUser = "SYSTEM"
$heartAction = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$heartCmd`""
$controlAction = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$controlCmd`""
$startup = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -RestartCount 100 -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable
Register-ScheduledTask -TaskName "Axon-Heart" -Action $heartAction -Trigger $startup -Settings $settings -User $taskUser -RunLevel Highest -Force | Out-Null
Register-ScheduledTask -TaskName "Axon-ControlPlane" -Action $controlAction -Trigger $startup -Settings $settings -User $taskUser -RunLevel Highest -Force | Out-Null

if ($PublicHost) {
    $caddy = Get-Command "caddy" -ErrorAction SilentlyContinue
    if (-not $caddy) {
        Write-Warning "PublicHost supplied, but Caddy is not installed. Install Caddy, rerun this script, and point DNS A/AAAA at this VM."
    } else {
        $caddyFile = Join-Path $opsRoot "Caddyfile"
        @"
$PublicHost {
    encode zstd gzip
    reverse_proxy 127.0.0.1:$ControlPort
}
"@ | Set-Content -Path $caddyFile -Encoding ascii
        $caddyAction = New-ScheduledTaskAction -Execute $caddy.Source -Argument "run --config `"$caddyFile`" --adapter caddyfile"
        Register-ScheduledTask -TaskName "Axon-HTTPS" -Action $caddyAction -Trigger $startup -Settings $settings -User $taskUser -RunLevel Highest -Force | Out-Null
        if (-not (Get-NetFirewallRule -DisplayName "Axon HTTPS" -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName "Axon HTTPS" -Direction Inbound -Protocol TCP -LocalPort 443 -Action Allow | Out-Null
        }
    }
}

Start-ScheduledTask -TaskName "Axon-Heart"
Start-Sleep -Seconds 2
Start-ScheduledTask -TaskName "Axon-ControlPlane"
if ($PublicHost -and (Get-ScheduledTask -TaskName "Axon-HTTPS" -ErrorAction SilentlyContinue)) {
    Start-ScheduledTask -TaskName "Axon-HTTPS"
}

Write-Host ""
Write-Host "Axon VM bootstrap complete."
Write-Host "Repository: $AxonRoot ($Branch)"
Write-Host "State root: $StateRoot"
Write-Host "Control token file: $tokenPath"
if ($PublicHost) {
    Write-Host "Axon Home URL: https://$PublicHost"
} else {
    Write-Host "Control plane is bound only to 127.0.0.1:$ControlPort until HTTPS is configured."
}
Write-Host ""
Write-Host "ONE-TIME PHONE ENROLLMENT TOKEN (store it in Axon Home, then do not paste it again):"
Write-Host $token
