param(
    [string]$PythonPath = 'D:\Users\杨云\AppData\Local\Programs\Python\Python311\python.exe',
    [switch]$FinalizeOnly,
    [switch]$CompilePdf
)
$ErrorActionPreference = 'Stop'
$StudyRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $StudyRoot
try {
    $StudyOutput = Join-Path $StudyRoot 'results\v3\relative_depth_validation_run2'
    $StudyComplete = Join-Path $StudyOutput 'COMPLETE.json'
    $StudyLock = Join-Path $StudyOutput '.running.lock'
    if (-not $FinalizeOnly) {
        if (Test-Path -LiteralPath $StudyLock) {
            Write-Output 'Training already has a running lock. Do not launch a duplicate batch.'
            exit 0
        }
        if (-not (Test-Path -LiteralPath $StudyComplete)) {
            $StudySavedModels = @(Get-ChildItem -LiteralPath (Join-Path $StudyOutput 'details') -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '_(hybrid|norel)\.json$' })
            if ($StudySavedModels.Count -ne 12) {
                & $PythonPath -u scripts/v3_depth_validation.py
                $StudySavedModels = @(Get-ChildItem -LiteralPath (Join-Path $StudyOutput 'details') -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '_(hybrid|norel)\.json$' })
                if ($StudySavedModels.Count -ne 12) { throw 'Training incomplete. Completed artifacts are retained.' }
            }
            if (-not (Test-Path -LiteralPath $StudyComplete)) {
                & $PythonPath -u scripts/v3_recover_boundary_inference.py
                if ($LASTEXITCODE -ne 0) { throw 'Saved-model boundary inference failed.' }
            }
        }
    } else {
        $StudyDeadline = (Get-Date).AddMinutes(60)
        while (-not (Test-Path -LiteralPath $StudyComplete)) {
            if (-not (Test-Path -LiteralPath $StudyLock)) { throw 'Training is no longer running and has not completed.' }
            if ((Get-Date) -gt $StudyDeadline) { throw 'Timed out waiting for the numerical phase.' }
            Start-Sleep -Seconds 15
        }
    }
    & $PythonPath scripts/v3_summarize_depth_validation.py *> results/v3/summary_execution.log
    if ($LASTEXITCODE -ne 0) { throw 'Summary failed; inspect results/v3/summary_execution.log' }
    & $PythonPath scripts/v3_build_manuscript.py
    if ($LASTEXITCODE -ne 0) { throw 'Manuscript source build failed.' }
    & $PythonPath scripts/v3_verify_revision.py *> results/v3/verification_execution.log
    if ($LASTEXITCODE -ne 0) { throw 'Artifact verification failed; inspect results/v3/verification_execution.log' }
    if ($CompilePdf) {
        $StudyCompiler = 'C:\Users\杨云\.codex\plugins\cache\openai-bundled\latex\0.2.8\scripts\compile_latex.py'
        foreach ($StudyDoc in @('main', 'supplementary')) {
            $StudyTex = Join-Path $StudyRoot "paper\gse_v3\$StudyDoc.tex"
            & $PythonPath $StudyCompiler $StudyTex --compiler texlive --json > "paper/gse_v3/$($StudyDoc)_build.json"
            if ($LASTEXITCODE -ne 0) { throw "PDF build failed: $StudyDoc" }
        }
    }
    Write-Output 'V3 numerical phase, summaries and manuscript sources completed. No OpenAI API calls are used.'
} finally {
    Pop-Location
}
