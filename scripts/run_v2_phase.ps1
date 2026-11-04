param(
    [string]$PythonPath = 'D:\Users\杨云\AppData\Local\Programs\Python\Python311\python.exe'
)
$ErrorActionPreference = 'Stop'
$StudyRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $StudyRoot
try {
    if (-not (Test-Path -LiteralPath $PythonPath)) { throw "Python not found: $PythonPath" }
    $StudyLock = Join-Path $StudyRoot 'results\v2\pilot_common_core\.running.lock'
    if (Test-Path -LiteralPath $StudyLock) {
        Write-Output 'Pilot already has a running lock. Read results/v2/pilot_common_core/progress.jsonl; do not start a duplicate batch.'
        exit 0
    }
    & $PythonPath -u scripts/v2_pilot_common_core.py
    if ($LASTEXITCODE -ne 0) { throw 'Pilot failed. Completed per-fit JSON files are retained for resume.' }
    & $PythonPath -u scripts/v2_mask_thread_control.py
    if ($LASTEXITCODE -ne 0) { throw 'Same-thread diagnostic failed.' }
    & $PythonPath -u scripts/v2_existing_well_analysis.py
    if ($LASTEXITCODE -ne 0) { throw 'Existing-well analysis failed.' }
    & $PythonPath -u scripts/v2_shift_decomposition.py
    if ($LASTEXITCODE -ne 0) { throw 'Shift decomposition failed.' }
    & $PythonPath -u scripts/v2_summarize_pilot.py
    if ($LASTEXITCODE -ne 0) { throw 'Pilot summary failed.' }
    & $PythonPath -u scripts/v2_build_manuscript.py
    if ($LASTEXITCODE -ne 0) { throw 'V2 manuscript source build failed.' }
    Write-Output 'v2 numerical phase and manuscript sources completed. No OpenAI API is used by these scripts.'
} finally {
    Pop-Location
}
