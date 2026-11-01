param(
    [ValidateSet('01_straight','02_diagonal','03_side_turn','04_far','05_close_spacing','06_long_drive','07_one_metre_drive')]
    [string]$Case = '02_diagonal',
    [switch]$AllCases,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [ValidateSet('amdgpu','cpu')][string]$Backend = 'amdgpu',
    [string]$PythonPath = 'python3',
    [switch]$NoVideo
)
$ErrorActionPreference = 'Stop'
if ($AllCases -and $PSBoundParameters.ContainsKey('Case')) { throw 'Choose -Case or -AllCases.' }
$taskOutput = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if (Test-Path -LiteralPath $taskOutput) { throw 'Output already exists; choose a new output directory.' }
$taskLinuxSource = & wsl.exe -d Ubuntu-24.04 -- wslpath -a $PSScriptRoot.Replace('\','/')
if ($LASTEXITCODE -ne 0) { throw 'Source path conversion failed.' }
$taskLinuxOutput = & wsl.exe -d Ubuntu-24.04 -- wslpath -a $taskOutput.Replace('\','/')
if ($LASTEXITCODE -ne 0) { throw 'Output path conversion failed.' }
$taskArguments = @('--output-dir',$taskLinuxOutput.Trim(),'--backend',$Backend)
if ($AllCases) { $taskArguments += '--all-cases' } else { $taskArguments += @('--case',$Case) }
if ($NoVideo) { $taskArguments += '--no-video' }
& wsl.exe -d Ubuntu-24.04 -- env "PHYSICAL_AI_PYTHON=$PythonPath" bash "$($taskLinuxSource.Trim())/run-wsl.sh" @taskArguments
if ($LASTEXITCODE -ne 0) { throw "Trial failed or did not complete. Inspect $taskOutput\suite-summary.json and process logs." }
