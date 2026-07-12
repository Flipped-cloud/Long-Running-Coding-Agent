$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
@'
def divide(a: float, b: float) -> float:
    return a * b
'@ | Set-Content -Path (Join-Path $root "calculator.py") -Encoding UTF8
Write-Host "toy repo reset"
