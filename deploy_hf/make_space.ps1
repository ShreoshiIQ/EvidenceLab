# Builds ..\tg_space = the minimal folder to push to a Hugging Face Space. Run from the project root:  .\deploy_hf\make_space.ps1
$dst = Join-Path (Split-Path -Parent (Get-Location)) "tg_space"
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory -Path $dst | Out-Null
foreach ($d in "agentic","evaluation","gateway","pipelines","tgdb","dashboard") {
  robocopy $d (Join-Path $dst $d) /E /XD __pycache__ /XF *.pyc | Out-Null
}
foreach ($f in "common.py","config.py","answer.py","requirements.txt") { Copy-Item $f $dst }
New-Item -ItemType Directory -Path (Join-Path $dst "runs") | Out-Null
Copy-Item "runs\public_*.jsonl" (Join-Path $dst "runs")          # public results only: the hidden questions stay private
New-Item -ItemType Directory -Path (Join-Path $dst "data") | Out-Null
Copy-Item "data\eval_public.jsonl" (Join-Path $dst "data")        # example questions for the Ask tab
Copy-Item "deploy_hf\Dockerfile" $dst
Copy-Item "deploy_hf\README.md" $dst
if (Test-Path (Join-Path $dst ".env")) { throw ".env must never be copied" }
Write-Host "Space folder ready: $dst"
Get-ChildItem $dst | Select-Object Name
