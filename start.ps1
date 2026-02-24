# start.ps1
# Script to launch the Semantic Energy Live Application

Write-Host "Starting Semantic Energy Live Application..." -ForegroundColor Cyan

# 1. Start the FastAPI Backend using the virtual environment
Write-Host "Starting Backend on http://127.0.0.1:8000..." -ForegroundColor Yellow
$backendProcess = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "app.py" -WorkingDirectory ".\backend" -PassThru -NoNewWindow

# Give the backend a few seconds to initialize
Start-Sleep -Seconds 3

# 2. Start the Frontend Simple HTTP Server
Write-Host "Starting Frontend on http://127.0.0.1:3000..." -ForegroundColor Green
$frontendProcess = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m http.server 3000" -WorkingDirectory ".\frontend" -PassThru -NoNewWindow

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "Application is running!" -ForegroundColor White
Write-Host "Frontend: http://127.0.0.1:3000" -ForegroundColor Green
Write-Host "Backend API: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop both servers." -ForegroundColor Red
Write-Host "========================================================`n" -ForegroundColor Cyan

try {
    # Keep the script running to hold the processes open
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    # Clean up the background processes when the user presses Ctrl+C
    Write-Host "`nShutting down servers..." -ForegroundColor Red
    if ($backendProcess -and !$backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force
    }
    if ($frontendProcess -and !$frontendProcess.HasExited) {
        Stop-Process -Id $frontendProcess.Id -Force
    }
    Write-Host "Shutdown complete." -ForegroundColor Green
}
