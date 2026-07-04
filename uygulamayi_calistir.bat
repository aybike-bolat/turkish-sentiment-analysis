@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  Turkce Duygu Analizi - Streamlit Baslatiliyor
echo ============================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo HATA: venv klasoru bulunamadi. Once kurulumu tamamlamaniz gerekiyor.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Tarayici birazdan otomatik acilacak...
echo Uygulamayi kapatmak icin bu pencereyi kapatabilir ya da CTRL+C yapabilirsiniz.
echo.

streamlit run app\streamlit_app.py

pause
