:: Windows batch script for chaining single-command asdc calls.

:: run 4 step-CV experiments...
for /l %%x in (1, 1, 4) do (
   python asdc/scripts/cli.py step --direction +x --no-lift --press --verbose
   python asdc/scripts/cli.py cv --cell EXTERNAL --verbose
)
