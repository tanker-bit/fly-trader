# D:\GitHub inventory (v2, 2026-11-09)

## What I can and cannot see
My workspace is a cloud sandbox that is NOT linked to your computer, so I cannot list `D:\GitHub`. Nothing below is guessed; it comes only from your pasted terminal output. The real tree has to come from your machine.

## Known (from your pastes)
```
D:\github\
└── fly-trader\                      OUTER folder (stray; from first zip extraction)
    ├── .venv\                       stray venv, unused, safe to delete
    └── fly-trader\                  THE REPOSITORY (git root; origin tanker-bit/fly-trader; branch main)
        ├── .git\  .venv\  .env (secret, gitignored)  .gitignore  README.md  requirements.txt
        ├── baselines\ broker\ connectome\ env\ experiments\ learning\ sim\ tests\
        ├── tournament\              league engine
        ├── evalkit\  reports\       evaluation suite (created later; push unconfirmed)
        ├── dashboard\  dashboard3d\ visual dashboards (push unconfirmed)
        ├── data\                    gitignored (mb_*.csv, raw parquet, panel.parquet)
        ├── data_fundamentals\       gitignored (panel.parquet with fundamentals)
        ├── league_data\             gitignored (older league run)
        └── league_data_fundamentals\ gitignored (agents, rounds, scores)
```
Python itself lives at `C:\Users\abdul\AppData\Local\Python\pythoncore-3.14-64\` (from a traceback path). Do not share that account name more widely than needed.

## Not known
Any other repository under `D:\GitHub`, duplicate/backup copies, the Downloads folder where chat files were saved, hidden files beyond the above.

## Get the real listing (one command, Command Prompt)
```
powershell -NoProfile -Command "Get-ChildItem D:\GitHub -Recurse -Force -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch '\\(\.venv|\.git|__pycache__|node_modules)(\\|$)' } | ForEach-Object { if ($_.PSIsContainer) { 'DIR   ' + $_.FullName } else { 'FILE  ' + $_.FullName + '   ' + $_.Length + ' bytes   ' + $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm') } } | Out-File -Encoding utf8 D:\d_drive_github_listing.txt"
```
(Untested by me. File names/sizes/dates only, no contents; skim it before attaching if other private folders live there. Fallback: `dir D:\GitHub /s /b /a > D:\d_drive_github_listing.txt`.) Attach the result and I or the next agent can complete this file.

## Which folder is the project
`D:\github\fly-trader\fly-trader`.
