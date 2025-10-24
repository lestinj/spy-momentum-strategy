@echo off
REM ========================================
REM ML-Adaptive V49 Auto-Start Setup
REM For Windows Task Scheduler
REM ========================================

echo ========================================
echo ML-ADAPTIVE V49 AUTO-START SETUP
echo ========================================
echo.

REM Create the startup batch file
echo Creating startup script...

(
echo @echo off
echo REM ML-Adaptive V49 Morning Startup
echo.
echo echo Starting ML-Adaptive V49 Alerts...
echo cd /d "C:\path\to\your\scripts"
echo.
echo REM Wait for network to be ready
echo ping -n 10 127.0.0.1 ^>nul
echo.
echo REM Start the Python script
echo python ml_adaptive_v49_alerts_fixed.py --once
echo.
echo REM Optional: Keep window open to see results
echo pause
) > ml_v49_morning_check.bat

echo ✓ Created ml_v49_morning_check.bat
echo.

REM Create the Task Scheduler XML
echo Creating Task Scheduler configuration...

(
echo ^<?xml version="1.0" encoding="UTF-16"?^>
echo ^<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"^>
echo   ^<RegistrationInfo^>
echo     ^<Date^>2025-10-20T00:00:00^</Date^>
echo     ^<Author^>ML-V49^</Author^>
echo     ^<Description^>Run ML-Adaptive V49 alerts every morning at market open^</Description^>
echo   ^</RegistrationInfo^>
echo   ^<Triggers^>
echo     ^<CalendarTrigger^>
echo       ^<StartBoundary^>2025-10-21T09:00:00^</StartBoundary^>
echo       ^<Enabled^>true^</Enabled^>
echo       ^<ScheduleByWeek^>
echo         ^<DaysOfWeek^>
echo           ^<Monday /^>
echo           ^<Tuesday /^>
echo           ^<Wednesday /^>
echo           ^<Thursday /^>
echo           ^<Friday /^>
echo         ^</DaysOfWeek^>
echo         ^<WeeksInterval^>1^</WeeksInterval^>
echo       ^</ScheduleByWeek^>
echo     ^</CalendarTrigger^>
echo   ^</Triggers^>
echo   ^<Principals^>
echo     ^<Principal id="Author"^>
echo       ^<LogonType^>InteractiveToken^</LogonType^>
echo       ^<RunLevel^>LeastPrivilege^</RunLevel^>
echo     ^</Principal^>
echo   ^</Principals^>
echo   ^<Settings^>
echo     ^<MultipleInstancesPolicy^>IgnoreNew^</MultipleInstancesPolicy^>
echo     ^<DisallowStartIfOnBatteries^>false^</DisallowStartIfOnBatteries^>
echo     ^<StopIfGoingOnBatteries^>false^</StopIfGoingOnBatteries^>
echo     ^<AllowHardTerminate^>true^</AllowHardTerminate^>
echo     ^<StartWhenAvailable^>true^</StartWhenAvailable^>
echo     ^<RunOnlyIfNetworkAvailable^>true^</RunOnlyIfNetworkAvailable^>
echo     ^<WakeToRun^>false^</WakeToRun^>
echo     ^<Enabled^>true^</Enabled^>
echo   ^</Settings^>
echo   ^<Actions Context="Author"^>
echo     ^<Exec^>
echo       ^<Command^>C:\path\to\your\scripts\ml_v49_morning_check.bat^</Command^>
echo     ^</Exec^>
echo   ^</Actions^>
echo ^</Task^>
) > ML_V49_Task.xml

echo ✓ Created ML_V49_Task.xml
echo.

echo ========================================
echo MANUAL SETUP INSTRUCTIONS
echo ========================================
echo.
echo 1. EDIT ml_v49_morning_check.bat:
echo    - Change "C:\path\to\your\scripts" to your actual path
echo    - Adjust python command if using virtual environment
echo.
echo 2. IMPORT TO TASK SCHEDULER:
echo    a. Open Task Scheduler (Win+R, type: taskschd.msc)
echo    b. Click "Import Task" in right panel
echo    c. Select ML_V49_Task.xml
echo    d. Update the path in Actions tab
echo    e. Click OK
echo.
echo 3. SCHEDULE OPTIONS:
echo    Current: 9:00 AM every weekday
echo    Modify in Task Scheduler if needed:
echo    - 9:00 AM = Before market open check
echo    - 9:30 AM = At market open
echo    - 3:30 PM = Near market close
echo.
echo 4. TEST IT:
echo    Right-click task and select "Run" to test
echo.
echo ========================================
pause
