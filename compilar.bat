@echo off
title Compilador - Acompanhamento Crescimento OMS
cd /d "%~dp0"

echo =======================================================
echo    COMPILADOR DE EXECUTAVEL - DIAGNOSTICO E GERACAO
echo =======================================================
echo.

:: 1. Localiza se existe python ou py no sistema
set "PYTHON_CMD="
where python >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=python"

if not defined PYTHON_CMD (
    where py >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py"
)

if defined PYTHON_CMD goto :python_encontrado

echo [ERRO] O Python nao foi encontrado no sistema ou nao esta no PATH!
echo.
echo Para corrigir:
echo 1. Abra o instalador do Python.
echo 2. Selecione Modificar ou reinstale.
echo 3. Marque a opcao: Add python.exe to PATH.
echo.
goto :final

:python_encontrado
echo [1/4] Python localizado: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: 2. Cria ou valida o venv
if exist "venv\Scripts\python.exe" goto :venv_ok

echo [2/4] Criando ambiente isolado venv...
%PYTHON_CMD% -m venv venv
if errorlevel 1 (
    echo [ERRO] Falha ao criar o ambiente virtual venv.
    goto :final
)
goto :instalar_deps

:venv_ok
echo [2/4] Ambiente virtual venv existente e valido.

:instalar_deps
echo.
echo [3/4] Atualizando pip e instalando bibliotecas...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install pandas numpy scipy matplotlib pyinstaller
if errorlevel 1 (
    echo [ERRO] Falha na instalacao das dependencias via pip.
    goto :final
)

:: 3. Geracao do executavel
echo.
echo [4/4] Gerando executavel standalone, aguarde alguns instantes...
venv\Scripts\pyinstaller.exe --noconsole --onefile --name "AcompanhamentoCrescimento" --clean app.py

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