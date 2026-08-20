param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Prompt
)

$ErrorActionPreference = "Stop"

$promptText = ($Prompt -join " ").Trim()
$scriptPath = Join-Path $PSScriptRoot "hermes_glm.py"
if (-not $promptText -and [Console]::IsInputRedirected) {
    $promptText = [Console]::In.ReadToEnd().Trim()
}

if (-not $promptText) {
    throw "Provide a prompt argument or pipe prompt text into this script."
}

& python $scriptPath $promptText
exit $LASTEXITCODE
