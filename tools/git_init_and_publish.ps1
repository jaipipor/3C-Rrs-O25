\
        <#
        This script helps you initialize a local repo and publish it to GitHub using
        `gh` (GitHub CLI) if available. It is optional — GitHub Desktop provides a
        UI for publishing.

        Usage:
          .\tools\git_init_and_publish.ps1 -repoName "3C-Rrs-O25" -owner "your-gh-username"
        #>
        param(
          [Parameter(Mandatory=$true)][string]$repoName,
          [Parameter(Mandatory=$true)][string]$owner
        )

        # ensure repo is initialized
        if (-not (Test-Path .git)) {
            git init
            git add .
            git commit -m "scaffold: initial commit"
        }

        # attempt to create remote using gh if available
        if (Get-Command gh -ErrorAction SilentlyContinue) {
            gh repo create $owner/$repoName --public --source=. --remote=origin --push
        } else {
            Write-Host "gh CLI not found. Please create a repo on GitHub and add remote 'origin' manually." -ForegroundColor Yellow
        }
