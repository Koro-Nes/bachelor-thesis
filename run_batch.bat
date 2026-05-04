@echo off

REM Loop through all passed arguments (seeds)
for %%s in (%*) do (
    echo Running seed %%s...
    call just wsl --cr 100 --seed %%s
    echo Finished seed %%s
)

echo All runs completed.
pause