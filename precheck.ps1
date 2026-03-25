param(
    [string]$ImageName = "openenv-support-triage",
    [string]$ContainerName = "openenv-support-triage-precheck",
    [int]$Port = 7860,
    [string]$Model = "gpt-4o-mini"
)

$ErrorActionPreference = "Stop"

function Assert-Condition {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

$openenvCommand = Get-Command openenv -ErrorAction SilentlyContinue
if ($null -ne $openenvCommand) {
    Write-Host "[0/5] Running openenv validate..." -ForegroundColor Cyan
    openenv validate | Out-Host
    Assert-Condition ($LASTEXITCODE -eq 0) "openenv validate failed"
}
else {
    Write-Host "[0/5] openenv CLI not found; skipping openenv validate" -ForegroundColor DarkYellow
}

Write-Host "[1/5] Building Docker image..." -ForegroundColor Cyan
docker build -t $ImageName . | Out-Host
Assert-Condition ($LASTEXITCODE -eq 0) "Docker build failed"

Write-Host "[2/5] Starting container..." -ForegroundColor Cyan
docker run --rm -d -p "${Port}:7860" --name $ContainerName $ImageName | Out-Host
Assert-Condition ($LASTEXITCODE -eq 0) "Failed to start container"

try {
    Start-Sleep -Seconds 2

    Write-Host "[3/5] Running endpoint checks..." -ForegroundColor Cyan

    $root = Invoke-RestMethod -Method Get -Uri "http://localhost:$Port/"
    Assert-Condition ($root.status -eq "ok") "Root endpoint check failed"

    $resetBody = '{"task_id":"support-easy-001"}'
    $reset = Invoke-RestMethod -Method Post -Uri "http://localhost:$Port/reset" -ContentType "application/json" -Body $resetBody
    Assert-Condition ($reset.task_id -eq "support-easy-001") "Reset endpoint returned unexpected task_id"

    $tasks = Invoke-RestMethod -Method Get -Uri "http://localhost:$Port/tasks"
    Assert-Condition ($tasks.tasks.Count -ge 3) "Tasks endpoint returned fewer than 3 tasks"

    $grader = Invoke-RestMethod -Method Get -Uri "http://localhost:$Port/grader"
    Assert-Condition (($grader.score -ge 0.0) -and ($grader.score -le 1.0)) "Grader score out of [0,1]"

    $baselineBody = @{ model = $Model } | ConvertTo-Json -Compress
    $baseline = Invoke-RestMethod -Method Post -Uri "http://localhost:$Port/baseline" -ContentType "application/json" -Body $baselineBody
    Assert-Condition ($baseline.task_results.Count -ge 3) "Baseline result missing task scores"
    Assert-Condition (($baseline.average_score -ge 0.0) -and ($baseline.average_score -le 1.0)) "Baseline average score out of [0,1]"

    Write-Host "[5/5] Precheck passed" -ForegroundColor Green
    $baseline | ConvertTo-Json -Depth 8 | Out-Host
}
finally {
    Write-Host "Stopping container..." -ForegroundColor Yellow
    docker stop $ContainerName | Out-Null
}
