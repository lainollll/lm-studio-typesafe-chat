param([string]$Python = "py")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$ttsPython = Join-Path $PSScriptRoot ".venv-chatterbox\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ttsPython)) {
    if ($Python -eq "py") { & $Python -3.11 -m venv .venv-chatterbox }
    else { & $Python -m venv .venv-chatterbox }
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 environment creation failed." }
}
& $ttsPython -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $ttsPython -m pip install chatterbox-tts==0.1.6
if ($LASTEXITCODE -ne 0) { throw "Chatterbox installation failed." }
& $ttsPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency verification failed." }
Write-Host "Project Chatterbox installed. Narration uses existing cached Turbo weights offline."
