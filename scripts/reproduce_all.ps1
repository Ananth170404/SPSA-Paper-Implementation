# Reproduce every figure of the paper (Windows PowerShell).
#   .\scripts\reproduce_all.ps1            # full protocol
#   .\scripts\reproduce_all.ps1 -Quick     # smoke test, a few minutes
param([switch]$Quick, [string]$Device = "auto")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$extra = @("--device", $Device)
if ($Quick) { $extra += "--quick" }

python scripts/check_setup.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/download_mnist.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/run_didactic.py @extra
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/run_mnist.py @extra
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/run_mnist.py @extra --order holistic --skip-gd --name mnist_holistic
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/run_mnist.py @extra --no-split-bias --skip-gd --name mnist_no_split_bias
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/run_mnist.py @extra --hidden 196 --name mnist_155k
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/run_mnist.py @extra --hidden 196 --order holistic --skip-gd --name mnist_155k_holistic
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/run_mnist.py @extra --hidden 196 --no-split-bias --skip-gd --name mnist_155k_no_split_bias
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not $Quick) { python experiments/summarize.py --out results/SUMMARY.md }
Write-Host "Done. Figures and SUMMARY.md are in results/ (results/quick/ for -Quick)."
