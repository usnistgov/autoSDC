:: Windows batch script for chaining single-command asdc calls.

:: run 4 step-CV experiments...
:: wait 10 seconds after each experiment is finished.
for /l %%x in (1, 1, 4) do (
   echo begin experiment %%x
   python asdc/scripts/cli.py step --direction +x --no-lift --press --verbose
   python asdc/scripts/cli.py cv --cell EXTERNAL --verbose
   echo finished collecting data
   timeout /t 10 /nobreak > NUL
)
