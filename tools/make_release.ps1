\
        param(
          [string]$tag = "v0.1.0",
          [string]$message = "First release"
        )

        # create tag and push
        git tag -a $tag -m $message
        git push origin $tag
        Write-Host "Tagged and pushed $tag" -ForegroundColor Green

        # Optionally create a GitHub Release using gh
        if (Get-Command gh -ErrorAction SilentlyContinue) {
            gh release create $tag -t $tag -n $message
        }
