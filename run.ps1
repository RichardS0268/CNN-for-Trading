# enumerate all .yml files in configs folder's subfolders
$files = Get-ChildItem -Path configs -Filter *.yml -Recurse -File
Write-Host "train starts at $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")"
foreach ($file in $files) {
    # get the path of the file
    $path = $file.FullName
    # log the path
    Write-Host $path
    # run the command
    python main.py $path

    git add .
    git commit -m "train on $($file) finished at $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")"
    git push
}
Write-Host "train finishes at $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")"

# After train
git add .
git commit -m "train finished at $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")"
git push
# # get auth use ./OneDriveUploader.exe -ms -a "url"
# # where url is get here: https://link.zhihu.com/?target=https%3A//login.microsoftonline.com/common/oauth2/v2.0/authorize%3Fclient_id%3D78d4dc35-7e46-42c6-9023-2d39314433a5%26response_type%3Dcode%26redirect_uri%3Dhttp%3A//localhost/onedrive-login%26response_mode%3Dquery%26scope%3Doffline_access%2520User.Read%2520Files.ReadWrite.All
# # access this web page and allow the app to access your onedrive, then copy the url that's it
# # after auth, run this command to upload files
# ./OneDriveUploader.exe -c ./auth.json -s "logs" -r "log"