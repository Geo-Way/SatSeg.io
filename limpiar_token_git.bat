@echo off
echo.
echo  GeoWay SatSeg - Limpiar token del historial Git
echo  =================================================
echo  ADVERTENCIA: Esto reescribe el historial completo.
echo  Haz un backup antes si tienes cambios importantes.
echo.
pause

echo.
echo [1/5] Instalando git-filter-repo...
pip install git-filter-repo 2>nul
where git-filter-repo >nul 2>&1
if errorlevel 1 (
  echo    Alternativa: usando BFG Repo Cleaner
  goto BFG
)

echo [2/5] Eliminando token del historial completo...
:: Reemplaza el token real con un placeholder en TODOS los commits
git filter-repo --replace-text replace-patterns.txt --force

goto FINISH

:BFG
echo [2/5] Descargando BFG...
curl -L https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar -o bfg.jar 2>nul
echo [3/5] Limpiando con BFG...
java -jar bfg.jar --replace-text replace-patterns.txt
git reflog expire --expire=now --all
git gc --prune=now --aggressive

:FINISH
echo.
echo [4/5] Forzando push al repositorio remoto...
git push origin --force --all
git push origin --force --tags

echo.
echo [5/5] Listo.
echo.
echo  IMPORTANTE: Genera un token HF NUEVO en:
echo  https://huggingface.co/settings/tokens
echo  El token expuesto ya esta comprometido - no lo reutilices.
echo.
pause
