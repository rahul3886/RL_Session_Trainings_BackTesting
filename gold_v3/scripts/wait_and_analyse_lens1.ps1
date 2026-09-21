param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$TrainErrLog,

    [Parameter(Mandatory = $true)]
    [string]$BaselineFinal,

    [Parameter(Mandatory = $true)]
    [string]$BaselineBest
)

Set-Location -LiteralPath $ProjectRoot

while (-not (Test-Path -LiteralPath $TrainErrLog)) {
    Start-Sleep -Seconds 10
}

while ($true) {
    if (Select-String -Path $TrainErrLog -Pattern 'Lens 1 training complete' -Quiet) {
        break
    }

    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like 'python*' -and $_.CommandLine -like '*main.py --mode train-lens1*'
    }

    if (-not $active) {
        exit 1
    }

    Start-Sleep -Seconds 60
}

python main.py --mode analyse-lens1
python scripts\post_retrain_lens1_analysis.py --baseline-final $BaselineFinal --baseline-best $BaselineBest
