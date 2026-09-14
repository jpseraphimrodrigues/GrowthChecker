@echo off
title Compilador - Acompanhamento Crescimento OMS
cd /d "%~dp0"

echo =======================================================
echo    COMPILADOR DE EXECUTAVEL - DIAGNOSTICO E GERACAO
echo =======================================================
echo.

:: 1. Verifica se o uv esta instalado
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERRO] O uv nao foi encontrado no PATH!
    echo Instale em https://docs.astral.sh/uv/getting-started/installation/
    goto :final
)

echo [1/4] uv localizado:
uv --version
echo.

:: 2. Sincroniza o ambiente e as dependencias do projeto
echo [2/4] Sincronizando dependencias com uv...
uv sync
if errorlevel 1 (
    echo [ERRO] Falha na sincronizacao das dependencias via uv.
    goto :final
)

:: 3. Geracao do executavel
echo.
echo [3/4] Gerando executavel standalone, aguarde alguns instantes...
uv run pyinstaller --noconsole --onefile --name "AcompanhamentoCrescimento" --clean app.py

echo.
if exist "dist\AcompanhamentoCrescimento.exe" (
    echo =======================================================
    echo [SUCESSO] Compilacao finalizada com exito!
    echo Arquivo disponivel em: dist\AcompanhamentoCrescimento.exe
    echo =======================================================
) else (
    echo [ERRO] Ocorreu uma falha e o executavel nao foi gerado.
)

:final
echo.
echo Pressione qualquer tecla para sair...
pause >nul
