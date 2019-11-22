@echo off
set currentDate=date
git pull
git add --all
git commit -m "last worked on %currentDate%"
git push
set /p temp="press enter to continue..."