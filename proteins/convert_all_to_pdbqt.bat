@echo off

REM --- Full path to MGLTools python.exe ---
set PYTHON="C:\Program Files (x86)\MGLTools-1.5.7\python.exe"

REM --- Full path to the prepare_receptor4.py script ---
set SCRIPT="C:\Program Files (x86)\MGLTools-1.5.7\MGLToolsPckgs\AutoDockTools\Utilities24\prepare_receptor4.py"

REM --- Go to the folder containing the PDB files ---
cd /d "C:\Users\srini\Desktop\docking_project\proteins"

echo Starting batch conversion...

for %%f in (*.pdb) do (
    echo Processing %%f ...
    %PYTHON% %SCRIPT% -r "%%f" -o "%%~nf.pdbqt" -A hydrogens
)

echo All conversions complete!
pause
