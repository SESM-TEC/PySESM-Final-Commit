param(
    [int[]]$StreamSteps = @(4),
    [int[]]$Dimensions = @(2),
    [string[]]$Methods = @("uniform", "kdtree"),
    [string]$PythonScript = ".\main_debug.py",
    [string]$OutputDir = ".\ram_profiles"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command mprof -ErrorAction SilentlyContinue)) {
    throw "mprof was not found. Install memory_profiler first: pip install memory_profiler"
}

if (-not (Test-Path $PythonScript)) {
    throw "Python script not found: $PythonScript"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$results = @()

Write-Host "Running RAM profiling per configuration"
Write-Host "  Dimensions: $($Dimensions -join ', ')"
Write-Host "  StreamSteps: $($StreamSteps -join ', ')"
Write-Host "  Methods: $($Methods -join ', ')"
Write-Host "  OutputDir: $OutputDir"

foreach ($dim in $Dimensions) {
    foreach ($step in $StreamSteps) {
        foreach ($method in $Methods) {
            $configDir = Join-Path $OutputDir ("dim_{0}\step_{1}\method_{2}" -f $dim, $step, $method)
            New-Item -ItemType Directory -Force -Path $configDir | Out-Null

            $datFile = Join-Path $configDir "profile.dat"
            $peakFile = Join-Path $configDir "peak.txt"
            $dimOverride = "dim=$dim"
            $stepOverride = "stream_steps=[$step]"
            $methodOverride = "methods_to_test=[$method]"

            Write-Host "`nRunning dim=$dim, step=$step, method=$method"
            & mprof run --include-children --output $datFile python $PythonScript $dimOverride $stepOverride $methodOverride
            if ($LASTEXITCODE -ne 0) {
                throw "Experiment failed for dim=$dim, step=$step, method=$method."
            }

            $peakOutput = (& mprof peak $datFile 2>&1 | Out-String).Trim()
            $peakOutput | Out-File -FilePath $peakFile -Encoding UTF8

            $peakMiB = $null
            if ($peakOutput -match "([0-9]+(?:\.[0-9]+)?)\s*MiB") {
                $peakMiB = [double]$matches[1]
            }

            $results += [PSCustomObject]@{
                dim = $dim
                stream_step = $step
                method = $method
                peak_mib = $peakMiB
                profile_file = $datFile
                peak_file = $peakFile
            }

            Write-Host "Peak RAM: $peakMiB MiB"
            Write-Host "Saved in: $configDir"
        }
    }
}

$summaryFile = Join-Path $OutputDir "ram_summary.csv"
$results | Export-Csv -Path $summaryFile -NoTypeInformation -Encoding UTF8

Write-Host "`n=== CONFIG SUMMARY (MiB) ==="
$results | Sort-Object dim, stream_step, method | Format-Table -AutoSize
Write-Host "`nSaved:"
Write-Host "  $summaryFile"
