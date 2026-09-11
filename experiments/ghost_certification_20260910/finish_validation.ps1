param([switch]$ResumeNativeChecks)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonPath = Join-Path $repoPath '.venv/Scripts/python.exe'
$previousPath = Get-Location
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

function Invoke-QualificationCheck {
    param([string]$Executable, [string[]]$Arguments, [string]$Log)
    Write-Output "Running $Log"
    & $Executable @Arguments *> (Join-Path $PSScriptRoot $Log)
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $PSScriptRoot $Log) -Tail 35
        throw "Qualification failed: $Log"
    }
    Write-Output "Passed $Log"
}

function Invoke-CompressedHeadlessCheck {
    param([string]$Executable, [string]$Log)
    $savedFactorization = $env:GHOST_CPU_FACTORIZATION
    try {
        $env:GHOST_CPU_FACTORIZATION = 'compressed'
        Invoke-QualificationCheck $Executable @('-m','unittest','test_experimental_headless','-q') $Log
    } finally {
        $env:GHOST_CPU_FACTORIZATION = $savedFactorization
    }
}

try {
    Set-Location -LiteralPath (Join-Path $repoPath 'tools/GHOST/tests')
    $regression = @('-m','unittest','test_compressed_path','test_solver_efficiency',
        'test_solver_compression','test_solver_matrix_pipeline','test_compact_multi_region',
        'test_experimental_cpu','test_thin_sheet','test_2d_co_polarized','test_memory_safety',
        'test_assembly_equivalence','test_assembly_audit_updates','test_hpc_runtime',
        'test_2d_capability_acceptance','-q')
    if (-not $ResumeNativeChecks) {
        Invoke-QualificationCheck $pythonPath $regression 'final-regression-tests.log'
        Invoke-CompressedHeadlessCheck $pythonPath 'final-compressed-headless-tests.log'
    }
    foreach ($runtime in @('python36','python36-np114-scipy100')) {
        $legacyPython = Join-Path (Split-Path $repoPath -Parent) "hpc-py36-validation/$runtime/python.exe"
        if (-not $ResumeNativeChecks -or $runtime -ne 'python36') {
            Invoke-QualificationCheck $legacyPython @('-m','unittest','test_compressed_path','-q') "$runtime-final-tests.log"
            Invoke-CompressedHeadlessCheck $legacyPython "$runtime-compressed-headless-tests.log"
        }
        Invoke-QualificationCheck $legacyPython @((Join-Path $PSScriptRoot 'test_native_queries.py'),'-q') "$runtime-native-tests.log"
    }
    Set-Location -LiteralPath $repoPath
    Invoke-QualificationCheck $pythonPath @((Join-Path $PSScriptRoot 'test_native_queries.py'),'-q') 'final-native-query-tests.log'
    Invoke-QualificationCheck $pythonPath @((Join-Path $PSScriptRoot 'public_probe.py'),'--materials','--certified','--factor','dense') 'final-materials-dense-cert.log'
    Invoke-QualificationCheck $pythonPath @((Join-Path $PSScriptRoot 'public_probe.py'),'--materials','--certified','--repeat','2') 'final-materials-compressed-cert.log'
    Invoke-QualificationCheck $pythonPath @((Join-Path $PSScriptRoot 'validate_results.py')) 'final-results-validation.log'
} finally {
    Set-Location -LiteralPath $previousPath
}
